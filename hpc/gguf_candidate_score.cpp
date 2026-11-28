#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int max_tokens = 192;

std::vector<llama_token> tokens(const llama_vocab * vocab, const std::string & text, bool special) {
    std::vector<llama_token> ids(max_tokens + 1);
    const auto count = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, special);
    if (count < 0 || count > max_tokens) {
        throw std::runtime_error("Candidate exceeds token cap");
    }
    ids.resize(count);
    return ids;
}

float score(llama_context * ctx, const std::vector<llama_token> & ids, int vocabulary_size, bool serial) {
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<int8_t> output(ids.size() - 1, 1);
    float total = 0.0f;
    const int step = serial ? 1 : output.size();
    for (int start = 0; start < static_cast<int>(output.size()); start += step) {
        auto batch = llama_batch_get_one(const_cast<llama_token *>(ids.data() + start), step);
        batch.logits = output.data() + start;
        if (llama_decode(ctx, batch) != 0) {
            throw std::runtime_error("Backend rejected the candidate sequence");
        }
        for (int i = 0; i < step; i++) {
            const auto * row = llama_get_logits_ith(ctx, i);
            if (!row) {
                throw std::runtime_error("Backend did not return the requested logit row");
            }
            const float peak = *std::max_element(row, row + vocabulary_size);
            float normalizer = 0.0f;
            for (int j = 0; j < vocabulary_size; j++) {
                normalizer += std::exp(row[j] - peak);
            }
            total += row[ids[start + i + 1]] - peak - std::log(normalizer);
        }
    }
    if (!std::isfinite(total)) {
        throw std::runtime_error("Nonfinite teacher-forced likelihood");
    }
    return total;
}

int main(int argc, char ** argv) try {
    if (argc != 2 || !std::getenv("SLURM_JOB_ID") || !std::getenv("RCAC_SCRATCH")) {
        throw std::runtime_error("Use a model path inside a Gilbreth allocation");
    }
    llama_backend_init();
    auto cuda = ggml_backend_dev_by_name("CUDA0");
    if (!cuda || !llama_supports_gpu_offload()) {
        throw std::runtime_error("Allocated CUDA backend is unavailable");
    }
    ggml_backend_dev_t devices[] = {cuda, nullptr};
    llama_model_tensor_buft_override overrides[] = {
        {"^token_embd\\.weight$", ggml_backend_dev_buffer_type(cuda)}, {nullptr, nullptr}
    };
    auto params = llama_model_default_params();
    params.devices = devices;
    params.tensor_buft_overrides = overrides;
    params.n_gpu_layers = -1;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], params), llama_model_free);
    if (!model) {
        throw std::runtime_error("Model loading failed");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    const auto delimiter = tokens(vocab, "<|endoftext|>", true);
    if (delimiter.size() != 1 || !llama_vocab_is_eog(vocab, delimiter[0])) {
        throw std::runtime_error("Unexpected Qwen tokenizer delimiter");
    }
    auto settings = llama_context_default_params();
    settings.n_ctx = 256;
    settings.n_batch = 256;
    settings.n_ubatch = 256;
    settings.n_threads = 4;
    settings.n_threads_batch = 4;
    settings.offload_kqv = true;
    settings.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), settings), llama_free);
    if (!ctx) {
        throw std::runtime_error("Could not allocate scoring context");
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto fixture = tokens(vocab, "The quick brown fox.", false);
    fixture.insert(fixture.begin(), delimiter[0]);
    fixture.push_back(delimiter[0]);
    const float single = score(ctx.get(), fixture, n_vocab, true);
    const float batch = score(ctx.get(), fixture, n_vocab, false);
    auto interrupt = tokens(vocab, "A separate candidate with different tokens.", false);
    interrupt.insert(interrupt.begin(), delimiter[0]);
    interrupt.push_back(delimiter[0]);
    const bool serial = std::abs(single - batch) > 0.05f;
    std::cerr << std::setprecision(9) << "scoring fixture nats (serial/batched): " << single << " / " << batch
              << "; selected mode: " << (serial ? "serial" : "batched") << '\n';
    score(ctx.get(), interrupt, n_vocab, serial);
    const float repeat = score(ctx.get(), fixture, n_vocab, serial);
    if (std::abs(repeat - (serial ? single : batch)) > 0.005f) {
        throw std::runtime_error("Fixture score changes after an unrelated candidate");
    }
    std::cout << std::setprecision(9) << delimiter[0] << ' ' << n_vocab << ' ' << single << ' ' << batch
              << ' ' << repeat << ' ' << (serial ? "serial" : "batched")
              << ' ' << llama_model_size(model.get()) << std::endl;
    for (std::string line; std::getline(std::cin, line);) {
        auto ids = tokens(vocab, line, false);
        ids.insert(ids.begin(), delimiter[0]);
        ids.push_back(delimiter[0]);
        const float value = score(ctx.get(), ids, n_vocab, serial);
        std::cout << value << ' ' << ids.size() - 1 << std::endl;
    }
    return 0;
} catch (const std::exception & exc) {
    std::cerr << exc.what() << '\n';
    return 1;
}
