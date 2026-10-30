// Debug tool (trace agent): causal prompt prefill + one Clover non-causal draft forward (anchor + masks), logits dumped
// to a binary file for CPU vs NPU comparison.
//   CLOVER_DUMP_OUT     output file (required)
//   CLOVER_ALL_LOGITS   1 = keep logits of every prompt row (perplexity-like causal pass), default: last row only
//   CLOVER_STATE_SAVE   save seq 0 state after the prefill
//   CLOVER_STATE_LOAD   skip the prefill and load seq 0 state instead (isolates the draft forward)
//   CLOVER_ANCHOR       anchor token id (default: argmax of the last prompt row)
//   CLOVER_DRAFT_N      draft block width incl. anchor (default 16, 0 = no draft)
//   CLOVER_TOKENS_FILE  raw int32 prompt token ids (instead of tokenizing -p/-f)
// File: int32 n_vocab, n_prompt, n_prefill_rows, n_draft, anchor; int32 tokens[n_prompt]; f32 rows (prefill then draft).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char * env(const char * name) {
    const char * v = std::getenv(name);
    return v && *v ? v : nullptr;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * out_path = env("CLOVER_DUMP_OUT");
    if (!out_path) {
        LOG_ERR("CLOVER_DUMP_OUT not set\n");
        return 1;
    }
    const bool   all_logits = env("CLOVER_ALL_LOGITS") && std::atoi(env("CLOVER_ALL_LOGITS")) != 0;
    const char * state_save = env("CLOVER_STATE_SAVE");
    const char * state_load = env("CLOVER_STATE_LOAD");
    const int    n_draft    = env("CLOVER_DRAFT_N") ? std::atoi(env("CLOVER_DRAFT_N")) : 16;

    llama_backend_init();
    llama_numa_init(params.numa);
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens;
    if (const char * tf = env("CLOVER_TOKENS_FILE")) {
        // raw int32 token ids (e.g. one perplexity chunk) instead of tokenizing the prompt
        FILE * f = std::fopen(tf, "rb");
        if (!f) {
            LOG_ERR("cannot open %s\n", tf);
            return 1;
        }
        llama_token t;
        while (std::fread(&t, sizeof(t), 1, f) == 1) {
            tokens.push_back(t);
        }
        std::fclose(f);
    } else {
        tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    }
    const int n_prompt = (int) tokens.size();
    if (n_prompt < 1 || n_prompt > (int) llama_n_ubatch(ctx)) {
        LOG_ERR("prompt has %d tokens; must be 1..n_ubatch (%d)\n", n_prompt, (int) llama_n_ubatch(ctx));
        return 1;
    }

    std::vector<float> rows;
    int n_prefill_rows = 0;
    if (state_load) {
        std::vector<llama_token> saved(n_prompt + 16);
        size_t n_saved = 0;
        if (llama_state_seq_load_file(ctx, state_load, 0, saved.data(), saved.size(), &n_saved) == 0 || (int) n_saved != n_prompt) {
            LOG_ERR("failed to load state %s (n_saved %zu)\n", state_load, n_saved);
            return 1;
        }
    } else {
        common_batch batch(ctx);
        for (int i = 0; i < n_prompt; ++i) {
            batch.add(tokens[i], i, 0, all_logits || i == n_prompt - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("prefill failed\n");
            return 1;
        }
        n_prefill_rows = all_logits ? n_prompt : 1;
        for (int r = 0; r < n_prefill_rows; ++r) {
            const float * l = llama_get_logits_ith(ctx, all_logits ? r : n_prompt - 1);
            rows.insert(rows.end(), l, l + n_vocab);
        }
        if (state_save && llama_state_seq_save_file(ctx, state_save, 0, tokens.data(), tokens.size()) == 0) {
            LOG_ERR("failed to save state %s\n", state_save);
            return 1;
        }
    }

    llama_token anchor = 0;
    if (env("CLOVER_ANCHOR")) {
        anchor = std::atoi(env("CLOVER_ANCHOR"));
    } else if (n_prefill_rows > 0) {
        const float * l = rows.data() + (size_t) (n_prefill_rows - 1) * n_vocab;
        anchor = (llama_token) (std::max_element(l, l + n_vocab) - l);
    } else {
        LOG_ERR("CLOVER_ANCHOR required with CLOVER_STATE_LOAD\n");
        return 1;
    }

    if (n_draft > 0) {
        const llama_token mask = llama_vocab_mask(vocab);
        if (mask == LLAMA_TOKEN_NULL) {
            LOG_ERR("model has no mask token\n");
            return 1;
        }
        common_batch batch(ctx);
        for (int i = 0; i < n_draft; ++i) {
            batch.add(i == 0 ? anchor : mask, n_prompt + i, 0, true);
        }
        llama_set_causal_attn(ctx, false);
        const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        llama_set_causal_attn(ctx, true);
        if (ret) {
            LOG_ERR("draft forward failed (%d)\n", ret);
            return 1;
        }
        for (int r = 0; r < n_draft; ++r) {
            const float * l = llama_get_logits_ith(ctx, r);
            rows.insert(rows.end(), l, l + n_vocab);
        }
    }

    FILE * f = std::fopen(out_path, "wb");
    if (!f) {
        LOG_ERR("cannot open %s\n", out_path);
        return 1;
    }
    const int32_t hdr[5] = { n_vocab, n_prompt, n_prefill_rows, std::max(n_draft, 0), anchor };
    std::fwrite(hdr, sizeof(int32_t), 5, f);
    std::fwrite(tokens.data(), sizeof(llama_token), tokens.size(), f);
    std::fwrite(rows.data(), sizeof(float), rows.size(), f);
    std::fclose(f);
    LOG_INF("clover-draft-dump: n_prompt %d prefill_rows %d draft %d anchor %d -> %s\n", n_prompt, n_prefill_rows, n_draft, anchor, out_path);

    llama_backend_free();
    return 0;
}
