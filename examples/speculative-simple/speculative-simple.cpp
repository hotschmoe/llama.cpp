#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }

        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);

    llama_tokens draft;

    common_prompt_checkpoint ckpt;

    // per-phase wall time (us) and forward counts; a re-verify is a verify of a draft rebuilt from a restored checkpoint
    struct {
        int64_t draft = 0, ckpt_save = 0, verify = 0, reverify = 0, plain = 0, sample = 0, ckpt_load = 0, seq_rm = 0;
        int32_t n_draft = 0, n_verify = 0, n_reverify = 0, n_plain = 0;
    } prof;
    bool draft_from_ckpt = false;

    // research knobs, off by default:
    // LLAMA_SPEC_BONUS=1: after a partial acceptance the next step is a plain 1-token decode of the correction token
    // LLAMA_SPEC_VERIFY_MAX=n: draft (and verify) at most n tokens; the draft-self block width stays n_max + 1
    const bool spec_bonus = getenv("LLAMA_SPEC_BONUS") && atoi(getenv("LLAMA_SPEC_BONUS")) != 0;
    const int  verify_max = getenv("LLAMA_SPEC_VERIFY_MAX") ? atoi(getenv("LLAMA_SPEC_VERIFY_MAX")) : -1;
    bool skip_draft = false;

    // optional dump of the committed token ids (one per line) for exactness checks
    FILE * f_tokens = nullptr;
    if (const char * path = getenv("LLAMA_SPEC_TOKENS_OUT")) {
        f_tokens = fopen(path, "w");
    }

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            if (verify_max >= 0) {
                n_draft_max = std::min(n_draft_max, verify_max);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            int64_t t0 = ggml_time_us();
            if (!skip_draft) {
                common_speculative_draft(spec);
            }
            skip_draft = false;
            int64_t t1 = ggml_time_us();
            prof.draft += t1 - t0;
            prof.n_draft += !draft.empty();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
            prof.ckpt_save += ggml_time_us() - t1;
            draft_from_ckpt = false;

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                batch_tgt.add(draft[i], n_past + i, seq_id, true);
            }


            const int64_t t0 = ggml_time_us();
            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
            llama_synchronize(ctx_tgt);
            const int64_t dt = ggml_time_us() - t0;
            if (draft_from_ckpt) {
                prof.reverify += dt;
                prof.n_reverify++;
            } else if (draft.empty()) {
                prof.plain += dt;
                prof.n_plain++;
            } else {
                prof.verify += dt;
                prof.n_verify++;
            }
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            break;
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const int64_t t_s = ggml_time_us();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        prof.sample += ggml_time_us() - t_s;

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            draft = std::move(ids);
            draft_from_ckpt = true;

            {
                const int64_t t0 = ggml_time_us();
                GGML_ASSERT(ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
                prof.ckpt_load += ggml_time_us() - t0;
            }

            if (ctx_dft) {
                GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        if (spec_bonus && !use_ckpt_tgt && n_draft > 0 && ids.size() - 1 < n_draft) {
            skip_draft = true;
        }

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (f_tokens) {
                fprintf(f_tokens, "%d\n", id_last);
            }

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            const int64_t t0 = ggml_time_us();
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
            prof.seq_rm += ggml_time_us() - t0;

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    if (f_tokens) {
        fclose(f_tokens);
    }

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    {
        const double t_dec = (t_dec_end - t_dec_start) / 1e3;
        const int n_fwd = prof.n_draft + prof.n_verify + prof.n_reverify + prof.n_plain;
        LOG_INF("\n");
        LOG_INF("spec prof: decode %.1f ms, %d tokens, forwards: draft %d, verify %d, re-verify %d, plain %d -> %.3f forwards/token\n",
                t_dec, n_predict, prof.n_draft, prof.n_verify, prof.n_reverify, prof.n_plain, n_predict > 0 ? (double) n_fwd / n_predict : 0.0);
        LOG_INF("spec prof: draft %.1f ms (%.2f/call), verify %.1f ms (%.2f/call), re-verify %.1f ms (%.2f/call)\n",
                prof.draft / 1e3, prof.draft / 1e3 / std::max(1, prof.n_draft),
                prof.verify / 1e3, prof.verify / 1e3 / std::max(1, prof.n_verify),
                prof.reverify / 1e3, prof.reverify / 1e3 / std::max(1, prof.n_reverify));
        LOG_INF("spec prof: plain 1-token decode %.1f ms (%.2f/call)\n", prof.plain / 1e3, prof.plain / 1e3 / std::max(1, prof.n_plain));
        LOG_INF("spec prof: ckpt save %.1f ms, ckpt load %.1f ms, sample %.1f ms, seq_rm %.1f ms, other %.1f ms\n",
                prof.ckpt_save / 1e3, prof.ckpt_load / 1e3, prof.sample / 1e3, prof.seq_rm / 1e3,
                t_dec - (prof.draft + prof.ckpt_save + prof.verify + prof.reverify + prof.plain + prof.sample + prof.ckpt_load + prof.seq_rm) / 1e3);
        LOG_INF("spec prof: tokens/step %.3f (steps = verify forwards of a fresh draft)\n", prof.n_verify > 0 ? (double) n_predict / prof.n_verify : 0.0);
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());


    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
