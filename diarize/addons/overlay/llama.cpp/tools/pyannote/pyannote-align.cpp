#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"
#include "nlohmann/json.hpp"
#include "pyannote-entrypoints.h"

using json = nlohmann::json;

struct diar_seg {
    double start_sec = 0.0;
    double end_sec = 0.0;
    std::string speaker = "UNKNOWN";
};

struct word_item {
    std::string word;
    double start_sec = 0.0;
    double end_sec = 0.0;
    int64_t start_token_index = -1;
    int64_t end_token_index = -1;
    std::string speaker = "UNKNOWN";
};

struct speaker_segment {
    std::string speaker = "UNKNOWN";
    double start_sec = 0.0;
    double end_sec = 0.0;
    int32_t num_words = 0;
    std::string text;
};

struct speaker_embedding {
    std::string speaker = "UNKNOWN";
    std::vector<float> values;
};

struct xvec_transform_model {
    std::vector<float> mean1;
    std::vector<float> mean2;
    std::vector<float> lda; // row-major [input_dim, output_dim]
    int32_t input_dim = 0;
    int32_t output_dim = 0;
};

struct plda_model {
    std::vector<float> mu;
    std::vector<float> tr;  // row-major [dim, dim]
    std::vector<float> psi;
    int32_t dim = 0;
};

struct plda_refinement_state {
    bool enabled = false;
    bool models_loaded = false;
    std::string status = "disabled";
    double sim_threshold = 0.55;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> pair_similarity;
    xvec_transform_model xvec;
    plda_model plda;
};

static void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " --words-json <path> --diarization-json <path> --out-prefix <path>\n"
        << "       [--max-gap-sec <float>] [--max-words <int>] [--max-duration-sec <float>]\n"
        << "       [--split-on-hard-break|--no-split-on-hard-break]\n"
        << "       [--plda-gguf <path>] [--xvec-transform-gguf <path>]\n"
        << "       [--plda-sim-threshold <float>] [--disable-plda-refinement]\n"
        << "       [--prefer-exclusive|--prefer-regular]\n\n"
        << "Reads word-level timestamps and pyannote diarization JSON, assigns speakers,\n"
        << "and writes:\n"
        << "  <out-prefix>.speaker_words.tsv\n"
        << "  <out-prefix>.speaker_transcript.txt\n"
        << "  <out-prefix>.speaker.srt\n"
        << "  <out-prefix>.speaker_alignment.json\n";
}

static std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static void write_text_file(const std::filesystem::path & path, const std::string & text) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    ofs << text;
}

static json read_json_file(const std::filesystem::path & path) {
    return json::parse(read_text_file(path));
}

static double round3(double x) {
    return std::round(x * 1000.0) / 1000.0;
}

static std::string fmt3(double x) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << x;
    return oss.str();
}

static std::string format_srt_time(double seconds) {
    seconds = std::max(0.0, seconds);
    const auto total_ms = static_cast<int64_t>(std::llround(seconds * 1000.0));
    int64_t rem = total_ms;
    const int64_t hours = rem / 3600000;
    rem -= hours * 3600000;
    const int64_t minutes = rem / 60000;
    rem -= minutes * 60000;
    const int64_t secs = rem / 1000;
    rem -= secs * 1000;
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << hours << ":"
        << std::setw(2) << minutes << ":"
        << std::setw(2) << secs << ","
        << std::setw(3) << rem;
    return oss.str();
}

static std::string format_hms(double seconds) {
    seconds = std::max(0.0, seconds);
    const int64_t total_s = static_cast<int64_t>(std::llround(seconds));
    int64_t rem = total_s;
    const int64_t hours = rem / 3600;
    rem -= hours * 3600;
    const int64_t minutes = rem / 60;
    rem -= minutes * 60;
    const int64_t secs = rem;
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << hours << ":"
        << std::setw(2) << minutes << ":"
        << std::setw(2) << secs;
    return oss.str();
}

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::string collapse_spaces(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = false;
    for (char ch : s) {
        const bool is_space = (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n');
        if (is_space) {
            if (!prev_space) {
                out.push_back(' ');
            }
            prev_space = true;
        } else {
            out.push_back(ch);
            prev_space = false;
        }
    }
    while (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

static std::string sanitize_tsv(const std::string & s) {
    std::string out = s;
    replace_all(out, "\t", " ");
    replace_all(out, "\r", " ");
    replace_all(out, "\n", " ");
    return collapse_spaces(out);
}

static std::string sanitize_word_token(std::string s) {
    replace_all(s, "<s>", "");
    replace_all(s, "</s>", "");
    replace_all(s, "[STREAMING_PAD]", "");
    replace_all(s, "[STREAMING_WORD]", "");
    replace_all(s, "---", "");
    s = collapse_spaces(s);
    return s;
}

static std::string join_words(const std::vector<std::string> & tokens) {
    if (tokens.empty()) {
        return "";
    }
    std::string text;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            text.push_back(' ');
        }
        text += tokens[i];
    }
    for (const char * punct : {".", ",", "?", "!", ";", ":", "%"}) {
        replace_all(text, std::string(" ") + punct, punct);
    }
    replace_all(text, " n't", "n't");
    replace_all(text, " '", "'");
    return collapse_spaces(text);
}

static double get_number(const json & obj, std::initializer_list<const char *> keys, double fallback) {
    for (const char * key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto & v = obj.at(key);
        if (v.is_number_float()) {
            return v.get<double>();
        }
        if (v.is_number_integer()) {
            return static_cast<double>(v.get<int64_t>());
        }
        if (v.is_string()) {
            try {
                return std::stod(v.get<std::string>());
            } catch (...) {
            }
        }
    }
    return fallback;
}

static std::string get_string(const json & obj, std::initializer_list<const char *> keys, const std::string & fallback = "") {
    for (const char * key : keys) {
        if (!obj.contains(key)) {
            continue;
        }
        const auto & v = obj.at(key);
        if (v.is_string()) {
            return v.get<std::string>();
        }
        if (v.is_number_float()) {
            return std::to_string(v.get<double>());
        }
        if (v.is_number_integer()) {
            return std::to_string(v.get<int64_t>());
        }
        if (v.is_boolean()) {
            return v.get<bool>() ? "true" : "false";
        }
    }
    return fallback;
}

static const json * find_words_array(const json & root) {
    if (root.is_array()) {
        return &root;
    }
    if (!root.is_object()) {
        return nullptr;
    }
    if (root.contains("timestamps") && root.at("timestamps").is_object()) {
        const auto & t = root.at("timestamps");
        if (t.contains("words") && t.at("words").is_array()) {
            return &t.at("words");
        }
    }
    if (root.contains("words") && root.at("words").is_array()) {
        return &root.at("words");
    }
    if (root.contains("speaker_words") && root.at("speaker_words").is_array()) {
        return &root.at("speaker_words");
    }
    return nullptr;
}

static std::vector<word_item> parse_words(const json & root) {
    const json * words_arr = find_words_array(root);
    if (words_arr == nullptr) {
        throw std::runtime_error("words JSON does not contain a words array");
    }
    std::vector<word_item> words;
    words.reserve(words_arr->size());
    for (const auto & it : *words_arr) {
        if (!it.is_object()) {
            continue;
        }
        word_item w;
        w.word = sanitize_word_token(get_string(it, {"word", "text"}, ""));
        if (w.word.empty()) {
            continue;
        }
        w.start_sec = get_number(it, {"start_sec", "start"}, 0.0);
        w.end_sec = get_number(it, {"end_sec", "end"}, w.start_sec);
        w.start_sec = std::max(0.0, w.start_sec);
        w.end_sec = std::max(w.start_sec, w.end_sec);
        if (it.contains("start_token_index") && it.at("start_token_index").is_number_integer()) {
            w.start_token_index = it.at("start_token_index").get<int64_t>();
        }
        if (it.contains("end_token_index") && it.at("end_token_index").is_number_integer()) {
            w.end_token_index = it.at("end_token_index").get<int64_t>();
        }
        if (it.contains("speaker")) {
            w.speaker = get_string(it, {"speaker"}, "UNKNOWN");
        }
        words.push_back(std::move(w));
    }
    std::sort(words.begin(), words.end(), [](const word_item & a, const word_item & b) {
        if (a.start_sec != b.start_sec) {
            return a.start_sec < b.start_sec;
        }
        return a.end_sec < b.end_sec;
    });
    return words;
}

static std::vector<diar_seg> parse_diar_segments_array(const json & arr) {
    std::vector<diar_seg> segs;
    if (!arr.is_array()) {
        return segs;
    }
    segs.reserve(arr.size());
    for (const auto & it : arr) {
        if (!it.is_object()) {
            continue;
        }
        diar_seg s;
        s.start_sec = get_number(it, {"start_sec", "start"}, 0.0);
        s.end_sec = get_number(it, {"end_sec", "end"}, s.start_sec);
        s.start_sec = std::max(0.0, s.start_sec);
        s.end_sec = std::max(s.start_sec, s.end_sec);
        s.speaker = get_string(it, {"speaker", "label"}, "UNKNOWN");
        if (s.speaker.empty()) {
            s.speaker = "UNKNOWN";
        }
        segs.push_back(std::move(s));
    }
    std::sort(segs.begin(), segs.end(), [](const diar_seg & a, const diar_seg & b) {
        if (a.start_sec != b.start_sec) {
            return a.start_sec < b.start_sec;
        }
        if (a.end_sec != b.end_sec) {
            return a.end_sec < b.end_sec;
        }
        return a.speaker < b.speaker;
    });
    return segs;
}

static std::pair<std::vector<diar_seg>, std::string> parse_diarization_segments(const json & root, bool prefer_exclusive) {
    const json * payload = &root;
    if (root.is_object() && root.contains("diarization_json") && root.at("diarization_json").is_object()) {
        payload = &root.at("diarization_json");
    }
    if (!payload->is_object()) {
        throw std::runtime_error("diarization JSON must be an object");
    }

    auto get_by_key = [&](const char * key) -> std::vector<diar_seg> {
        if (!payload->contains(key)) {
            return {};
        }
        return parse_diar_segments_array(payload->at(key));
    };

    std::vector<diar_seg> exclusive = get_by_key("exclusive_speaker_diarization");
    if (exclusive.empty()) {
        exclusive = get_by_key("exclusive_diarization");
    }
    std::vector<diar_seg> regular = get_by_key("regular_speaker_diarization");
    if (regular.empty()) {
        regular = get_by_key("speaker_diarization");
    }
    if (regular.empty()) {
        regular = get_by_key("diarization");
    }

    if (prefer_exclusive && !exclusive.empty()) {
        return {exclusive, "exclusive"};
    }
    if (!regular.empty()) {
        return {regular, "regular"};
    }
    if (!exclusive.empty()) {
        return {exclusive, "exclusive"};
    }
    return {{}, "none"};
}

static std::vector<speaker_embedding> parse_speaker_embeddings(const json & root) {
    const json * payload = &root;
    if (root.is_object() && root.contains("diarization_json") && root.at("diarization_json").is_object()) {
        payload = &root.at("diarization_json");
    }
    if (!payload->is_object() || !payload->contains("speaker_embeddings")) {
        return {};
    }

    const auto & emb_root = payload->at("speaker_embeddings");
    const json * speakers_arr = nullptr;
    if (emb_root.is_object() && emb_root.contains("speakers") && emb_root.at("speakers").is_array()) {
        speakers_arr = &emb_root.at("speakers");
    } else if (emb_root.is_array()) {
        speakers_arr = &emb_root;
    }
    if (speakers_arr == nullptr) {
        return {};
    }

    std::vector<speaker_embedding> out;
    out.reserve(speakers_arr->size());
    for (const auto & it : *speakers_arr) {
        if (!it.is_object()) {
            continue;
        }
        const std::string speaker = get_string(it, {"speaker", "label"}, "");
        if (speaker.empty()) {
            continue;
        }
        if (!it.contains("embedding") || !it.at("embedding").is_array()) {
            continue;
        }
        const auto & arr = it.at("embedding");
        speaker_embedding e;
        e.speaker = speaker;
        e.values.reserve(arr.size());
        for (const auto & v : arr) {
            if (v.is_number_float()) {
                e.values.push_back(static_cast<float>(v.get<double>()));
            } else if (v.is_number_integer()) {
                e.values.push_back(static_cast<float>(v.get<int64_t>()));
            }
        }
        if (!e.values.empty()) {
            out.push_back(std::move(e));
        }
    }
    return out;
}

static double dot_product(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return s;
}

static void l2_normalize(std::vector<float> & x) {
    double n2 = 0.0;
    for (float v : x) {
        n2 += static_cast<double>(v) * static_cast<double>(v);
    }
    if (n2 <= 0.0) {
        return;
    }
    const double inv = 1.0 / std::sqrt(n2);
    for (float & v : x) {
        v = static_cast<float>(static_cast<double>(v) * inv);
    }
}

static std::vector<float> gguf_load_tensor_f32(
    const std::filesystem::path & path,
    const std::string & tensor_name) {
    ggml_context * tctx = nullptr;
    gguf_init_params params = {};
    params.no_alloc = false;
    params.ctx = &tctx;
    gguf_context * gctx = gguf_init_from_file(path.string().c_str(), params);
    if (gctx == nullptr || tctx == nullptr) {
        if (gctx != nullptr) {
            gguf_free(gctx);
        }
        throw std::runtime_error("failed to load gguf: " + path.string());
    }

    std::vector<float> out;
    const ggml_tensor * t = ggml_get_tensor(tctx, tensor_name.c_str());
    if (t == nullptr) {
        gguf_free(gctx);
        ggml_free(tctx);
        throw std::runtime_error("tensor not found in gguf: " + tensor_name + " from " + path.string());
    }
    if (t->type != GGML_TYPE_F32) {
        gguf_free(gctx);
        ggml_free(tctx);
        throw std::runtime_error("tensor must be F32: " + tensor_name + " from " + path.string());
    }

    const size_t n = ggml_nelements(t);
    out.resize(n);
    const float * src = static_cast<const float *>(t->data);
    std::copy(src, src + n, out.begin());

    gguf_free(gctx);
    ggml_free(tctx);
    return out;
}

static xvec_transform_model load_xvec_transform_model(const std::filesystem::path & path) {
    xvec_transform_model x;
    x.mean1 = gguf_load_tensor_f32(path, "pyannote.xvec_transform.mean1");
    x.mean2 = gguf_load_tensor_f32(path, "pyannote.xvec_transform.mean2");
    x.lda = gguf_load_tensor_f32(path, "pyannote.xvec_transform.lda");

    const size_t in_dim = x.mean1.size();
    const size_t out_dim = x.mean2.size();
    if (in_dim == 0 || out_dim == 0) {
        throw std::runtime_error("invalid xvec transform dimensions");
    }
    if (x.lda.size() != in_dim * out_dim) {
        throw std::runtime_error("xvec lda size mismatch");
    }
    x.input_dim = static_cast<int32_t>(in_dim);
    x.output_dim = static_cast<int32_t>(out_dim);
    return x;
}

static plda_model load_plda_model(const std::filesystem::path & path) {
    plda_model p;
    p.mu = gguf_load_tensor_f32(path, "pyannote.plda.mu");
    p.tr = gguf_load_tensor_f32(path, "pyannote.plda.tr");
    p.psi = gguf_load_tensor_f32(path, "pyannote.plda.psi");

    const size_t dim = p.mu.size();
    if (dim == 0 || p.psi.size() != dim || p.tr.size() != dim * dim) {
        throw std::runtime_error("invalid plda tensor dimensions");
    }
    p.dim = static_cast<int32_t>(dim);
    return p;
}

static std::vector<float> apply_xvec_transform(
    const xvec_transform_model & xvec,
    const std::vector<float> & emb) {
    if (emb.size() != static_cast<size_t>(xvec.input_dim)) {
        return {};
    }
    std::vector<float> out(static_cast<size_t>(xvec.output_dim), 0.0f);
    for (int32_t j = 0; j < xvec.output_dim; ++j) {
        double acc = 0.0;
        for (int32_t i = 0; i < xvec.input_dim; ++i) {
            const float centered = emb[static_cast<size_t>(i)] - xvec.mean1[static_cast<size_t>(i)];
            const float w = xvec.lda[static_cast<size_t>(i) * static_cast<size_t>(xvec.output_dim) + static_cast<size_t>(j)];
            acc += static_cast<double>(centered) * static_cast<double>(w);
        }
        out[static_cast<size_t>(j)] = static_cast<float>(acc) - xvec.mean2[static_cast<size_t>(j)];
    }
    return out;
}

static std::vector<float> apply_plda_projection(
    const plda_model & plda,
    const std::vector<float> & x) {
    if (x.size() != static_cast<size_t>(plda.dim)) {
        return {};
    }
    std::vector<float> out(static_cast<size_t>(plda.dim), 0.0f);
    for (int32_t r = 0; r < plda.dim; ++r) {
        double acc = 0.0;
        for (int32_t c = 0; c < plda.dim; ++c) {
            const float v = x[static_cast<size_t>(c)] - plda.mu[static_cast<size_t>(c)];
            const float w = plda.tr[static_cast<size_t>(r) * static_cast<size_t>(plda.dim) + static_cast<size_t>(c)];
            acc += static_cast<double>(w) * static_cast<double>(v);
        }
        const float psi = std::max(1e-6f, plda.psi[static_cast<size_t>(r)]);
        out[static_cast<size_t>(r)] = static_cast<float>(acc / std::sqrt(static_cast<double>(psi)));
    }
    l2_normalize(out);
    return out;
}

static plda_refinement_state build_plda_refinement_state(
    const std::filesystem::path & plda_gguf,
    const std::filesystem::path & xvec_gguf,
    double sim_threshold,
    const std::vector<speaker_embedding> & speaker_embeddings,
    bool enabled) {
    plda_refinement_state st;
    st.enabled = enabled;
    st.sim_threshold = sim_threshold;
    if (!enabled) {
        st.status = "disabled";
        return st;
    }
    if (plda_gguf.empty() || xvec_gguf.empty()) {
        st.status = "missing_paths";
        return st;
    }
    if (!std::filesystem::exists(plda_gguf) || !std::filesystem::exists(xvec_gguf)) {
        st.status = "gguf_not_found";
        return st;
    }
    if (speaker_embeddings.size() < 2) {
        st.status = "insufficient_speaker_embeddings";
        return st;
    }

    st.xvec = load_xvec_transform_model(xvec_gguf);
    st.plda = load_plda_model(plda_gguf);
    if (st.xvec.output_dim != st.plda.dim) {
        throw std::runtime_error("xvec/plda dimensionality mismatch");
    }

    std::unordered_map<std::string, std::vector<float>> proj;
    for (const auto & se : speaker_embeddings) {
        auto x = apply_xvec_transform(st.xvec, se.values);
        if (x.empty()) {
            continue;
        }
        auto y = apply_plda_projection(st.plda, x);
        if (!y.empty()) {
            proj[se.speaker] = std::move(y);
        }
    }

    for (const auto & a : proj) {
        auto & row = st.pair_similarity[a.first];
        row[a.first] = 1.0;
        for (const auto & b : proj) {
            if (a.first == b.first) {
                continue;
            }
            row[b.first] = dot_product(a.second, b.second);
        }
    }

    st.models_loaded = !st.pair_similarity.empty();
    st.status = st.models_loaded ? "ok" : "no_valid_projected_embeddings";
    return st;
}

static double plda_similarity(
    const plda_refinement_state & st,
    const std::string & a,
    const std::string & b) {
    if (!st.models_loaded) {
        return 0.0;
    }
    auto it = st.pair_similarity.find(a);
    if (it == st.pair_similarity.end()) {
        return 0.0;
    }
    auto it2 = it->second.find(b);
    if (it2 == it->second.end()) {
        return 0.0;
    }
    return it2->second;
}

static bool plda_merge_ok(
    const plda_refinement_state & st,
    const std::string & a,
    const std::string & b) {
    if (!st.enabled || !st.models_loaded) {
        return true;
    }
    return plda_similarity(st, a, b) >= st.sim_threshold;
}

static double interval_overlap(double a_start, double a_end, double b_start, double b_end) {
    return std::max(0.0, std::min(a_end, b_end) - std::max(a_start, b_start));
}

static std::string pick_speaker_for_span(
    double start_sec,
    double end_sec,
    const std::vector<diar_seg> & diarization_segments) {
    std::string best_speaker = "UNKNOWN";
    double best_overlap = 0.0;
    double best_center_distance = std::numeric_limits<double>::infinity();
    const double center = 0.5 * (start_sec + end_sec);
    const double duration = std::max(1e-3, end_sec - start_sec);
    const double pad = std::min(0.08, 0.25 * duration);
    const double span_start = std::max(0.0, start_sec - pad);
    const double span_end = end_sec + pad;

    for (const auto & seg : diarization_segments) {
        const double overlap = interval_overlap(span_start, span_end, seg.start_sec, seg.end_sec);
        const double seg_center = 0.5 * (seg.start_sec + seg.end_sec);
        const double center_distance = std::abs(seg_center - center);
        if (overlap > best_overlap + 1e-9) {
            best_overlap = overlap;
            best_center_distance = center_distance;
            best_speaker = seg.speaker;
            continue;
        }
        if (std::abs(overlap - best_overlap) <= 1e-9 && center_distance < best_center_distance) {
            best_center_distance = center_distance;
            best_speaker = seg.speaker;
        }
    }

    if (best_overlap > 0.0) {
        return best_speaker;
    }

    for (const auto & seg : diarization_segments) {
        const double seg_center = 0.5 * (seg.start_sec + seg.end_sec);
        const double center_distance = std::abs(seg_center - center);
        if (center_distance < best_center_distance) {
            best_center_distance = center_distance;
            best_speaker = seg.speaker;
        }
    }
    return best_speaker;
}

static std::vector<word_item> assign_speakers_to_words(
    const std::vector<word_item> & words,
    const std::vector<diar_seg> & diarization_segments,
    const plda_refinement_state & plda_state) {
    std::vector<word_item> out = words;
    if (diarization_segments.empty()) {
        for (auto & w : out) {
            w.speaker = "UNKNOWN";
        }
        return out;
    }

    for (auto & w : out) {
        w.speaker = pick_speaker_for_span(w.start_sec, w.end_sec, diarization_segments);
    }

    // Smooth short speaker flips caused by timestamp jitter around turn boundaries.
    if (out.size() >= 3) {
        for (size_t i = 1; i + 1 < out.size(); ++i) {
            const auto & prev = out[i - 1];
            const auto & next = out[i + 1];
            auto & cur = out[i];
            const double dur = std::max(0.0, cur.end_sec - cur.start_sec);
            const double gap_prev = std::max(0.0, cur.start_sec - prev.end_sec);
            const double gap_next = std::max(0.0, next.start_sec - cur.end_sec);
            if (prev.speaker == next.speaker &&
                cur.speaker != prev.speaker &&
                dur <= 0.45 &&
                gap_prev <= 0.20 &&
                gap_next <= 0.20 &&
                plda_merge_ok(plda_state, cur.speaker, prev.speaker)) {
                cur.speaker = prev.speaker;
            }
        }
    }

    if (out.size() >= 4) {
        for (size_t i = 1; i + 2 < out.size(); ++i) {
            auto & w0 = out[i];
            auto & w1 = out[i + 1];
            const auto & left = out[i - 1];
            const auto & right = out[i + 2];
            const double run_dur = std::max(0.0, w1.end_sec - w0.start_sec);
            if (w0.speaker == w1.speaker &&
                left.speaker == right.speaker &&
                w0.speaker != left.speaker &&
                run_dur <= 0.75 &&
                plda_merge_ok(plda_state, w0.speaker, left.speaker)) {
                w0.speaker = left.speaker;
                w1.speaker = left.speaker;
            }
        }
    }

    auto is_terminal_word = [](const std::string & w) {
        for (int i = (int) w.size() - 1; i >= 0; --i) {
            const unsigned char c = (unsigned char) w[(size_t) i];
            if (std::isspace(c)) {
                continue;
            }
            if (c == '.' || c == '?' || c == '!') {
                return true;
            }
            if (std::isalnum(c)) {
                return false;
            }
        }
        return false;
    };

    auto starts_with_lower_alpha = [](const std::string & w) {
        for (char ch : w) {
            const unsigned char c = (unsigned char) ch;
            if (std::isalpha(c)) {
                return std::islower(c) != 0;
            }
        }
        return false;
    };

    auto starts_with_upper_alpha = [](const std::string & w) {
        for (char ch : w) {
            const unsigned char c = (unsigned char) ch;
            if (std::isalpha(c)) {
                return std::isupper(c) != 0;
            }
        }
        return false;
    };

    auto is_punct_only = [](const std::string & w) {
        bool has = false;
        for (char ch : w) {
            const unsigned char c = (unsigned char) ch;
            if (std::isspace(c)) {
                continue;
            }
            has = true;
            if (std::isalnum(c)) {
                return false;
            }
        }
        return has;
    };

    // Keep punctuation marks on the same speaker as neighboring lexical tokens.
    if (!out.empty()) {
        for (size_t i = 0; i < out.size(); ++i) {
            if (!is_punct_only(out[i].word)) {
                continue;
            }
            if (i > 0 && !is_punct_only(out[i - 1].word)) {
                out[i].speaker = out[i - 1].speaker;
                continue;
            }
            if (i + 1 < out.size() && !is_punct_only(out[i + 1].word)) {
                out[i].speaker = out[i + 1].speaker;
            }
        }
    }

    struct run_span {
        size_t begin = 0;
        size_t end = 0;
        std::string speaker;
    };

    auto build_runs = [&](const std::vector<word_item> & ws) {
        std::vector<run_span> runs;
        if (ws.empty()) {
            return runs;
        }
        size_t begin = 0;
        for (size_t i = 1; i <= ws.size(); ++i) {
            if (i == ws.size() || ws[i].speaker != ws[begin].speaker) {
                runs.push_back(run_span{begin, i - 1, ws[begin].speaker});
                begin = i;
            }
        }
        return runs;
    };

    auto count_tail_without_terminal = [&](size_t begin, size_t end) {
        int n = 0;
        for (size_t i = end + 1; i > begin; --i) {
            const size_t idx = i - 1;
            if (is_terminal_word(out[idx].word)) {
                break;
            }
            n += 1;
        }
        return n;
    };

    auto count_prefix_until_terminal = [&](size_t begin, size_t end) {
        int n = 0;
        for (size_t i = begin; i <= end; ++i) {
            n += 1;
            if (is_terminal_word(out[i].word)) {
                break;
            }
        }
        return n;
    };

    // Pragmatic punctuation-aware reassignment around speaker boundaries.
    for (int pass = 0; pass < 3; ++pass) {
        bool changed = false;
        auto runs = build_runs(out);
        if (runs.size() < 2) {
            break;
        }

        for (size_t r = 0; r + 1 < runs.size(); ++r) {
            const auto a = runs[r];
            const auto b = runs[r + 1];
            if (a.speaker == b.speaker) {
                continue;
            }

            if (is_terminal_word(out[a.end].word)) {
                continue;
            }

            const int a_tail = count_tail_without_terminal(a.begin, a.end);
            const int b_prefix = count_prefix_until_terminal(b.begin, b.end);
            const bool b_prefix_has_terminal = is_terminal_word(out[b.begin + (size_t) std::max(0, b_prefix - 1)].word);
            const int b_len = (int) (b.end - b.begin + 1);
            const int b_suffix = std::max(0, b_len - b_prefix);

            // Case 0: move a short sentence-closing prefix from next run to previous run.
            // Example: "... what are your thoughts on" + "this. Well, ..."
            if (!is_terminal_word(out[a.end].word) &&
                b_prefix_has_terminal &&
                b_prefix >= 1 && b_prefix <= 3 &&
                b_suffix >= 3) {
                if (b_prefix > 2 && !plda_merge_ok(plda_state, a.speaker, b.speaker)) {
                    continue;
                }
                const size_t end_move = b.begin + (size_t) b_prefix - 1;
                for (size_t i = b.begin; i <= end_move; ++i) {
                    out[i].speaker = a.speaker;
                }
                changed = true;
                continue;
            }

            // Case 1: previous speaker ends with 1-3 dangling words; next speaker has longer continuation.
            if (starts_with_lower_alpha(out[b.begin].word) &&
                a_tail >= 1 && a_tail <= 3 && b_prefix > a_tail) {
                if (a_tail > 1 && !plda_merge_ok(plda_state, a.speaker, b.speaker)) {
                    continue;
                }
                const size_t start = a.end + 1 - (size_t) a_tail;
                for (size_t i = start; i <= a.end; ++i) {
                    out[i].speaker = b.speaker;
                }
                changed = true;
                continue;
            }

            // Case 2: next speaker starts with short lowercase continuation that quickly ends a sentence.
            if (starts_with_lower_alpha(out[b.begin].word) &&
                b_prefix_has_terminal && b_prefix >= 1 && b_prefix <= 2) {
                if (b_prefix > 2 && !plda_merge_ok(plda_state, a.speaker, b.speaker)) {
                    continue;
                }
                const size_t end_move = b.begin + (size_t) b_prefix - 1;
                for (size_t i = b.begin; i <= end_move; ++i) {
                    out[i].speaker = a.speaker;
                }
                changed = true;
                continue;
            }

            // Case 3: first word of next run ends a sentence that likely belongs to previous run.
            if (is_terminal_word(out[b.begin].word) &&
                !is_terminal_word(out[a.end].word) &&
                b.begin < b.end &&
                starts_with_upper_alpha(out[b.begin + 1].word)) {
                out[b.begin].speaker = a.speaker;
                changed = true;
                continue;
            }
        }

        if (!changed) {
            break;
        }
    }

    return out;
}
static bool ends_with_hard_break(const std::string & s) {
    if (s.empty()) {
        return false;
    }
    const char c = s.back();
    return c == '.' || c == '?' || c == '!';
}

static std::vector<speaker_segment> build_speaker_segments(
    const std::vector<word_item> & speaker_words,
    double max_gap_sec,
    int max_words,
    double max_duration_sec,
    bool split_on_hard_break) {
    std::vector<speaker_segment> out;
    if (speaker_words.empty()) {
        return out;
    }

    std::vector<word_item> bucket;
    bucket.reserve(static_cast<size_t>(std::max(8, max_words)));

    auto flush_bucket = [&]() {
        if (bucket.empty()) {
            return;
        }
        std::vector<std::string> tokens;
        tokens.reserve(bucket.size());
        for (const auto & w : bucket) {
            tokens.push_back(w.word);
        }
        speaker_segment seg;
        seg.speaker = bucket.front().speaker;
        seg.start_sec = round3(bucket.front().start_sec);
        seg.end_sec = round3(bucket.back().end_sec);
        seg.num_words = static_cast<int32_t>(bucket.size());
        seg.text = join_words(tokens);
        out.push_back(std::move(seg));
        bucket.clear();
    };

    for (const auto & w : speaker_words) {
        if (bucket.empty()) {
            bucket.push_back(w);
            continue;
        }

        const auto & last = bucket.back();
        const double candidate_gap = w.start_sec - last.end_sec;
        const double candidate_duration = w.end_sec - bucket.front().start_sec;
        const bool speaker_changed = w.speaker != last.speaker;
        const bool hard_break = split_on_hard_break && ends_with_hard_break(last.word);
        const bool gap_break = max_gap_sec >= 0.0 && candidate_gap > max_gap_sec;
        const bool max_words_break = max_words > 0 && static_cast<int>(bucket.size()) >= max_words;
        const bool max_duration_break = max_duration_sec > 0.0 && candidate_duration > max_duration_sec;

        if (speaker_changed || gap_break || max_words_break || max_duration_break || hard_break) {
            flush_bucket();
            bucket.push_back(w);
            continue;
        }
        bucket.push_back(w);
    }
    flush_bucket();
    return out;
}

static std::string segments_to_srt(const std::vector<speaker_segment> & segments) {
    std::ostringstream oss;
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto & seg = segments[i];
        oss << (i + 1) << "\n";
        oss << format_srt_time(seg.start_sec) << " --> " << format_srt_time(seg.end_sec) << "\n";
        oss << seg.speaker << ": " << seg.text << "\n\n";
    }
    return oss.str();
}

static std::string segments_to_speaker_text(const std::vector<speaker_segment> & segments) {
    std::ostringstream oss;
    for (const auto & seg : segments) {
        oss << "[" << format_hms(seg.start_sec) << "-" << format_hms(seg.end_sec) << "] "
            << seg.speaker << ": " << seg.text << "\n";
    }
    return oss.str();
}

int llama_pyannote_align_main(int argc, char ** argv) {
    std::filesystem::path words_json_path;
    std::filesystem::path diarization_json_path;
    std::filesystem::path out_prefix;
    double max_gap_sec = -1.0;
    int max_words = 0;
    double max_duration_sec = -1.0;
    bool prefer_exclusive = true;
    bool split_on_hard_break = false;
    std::filesystem::path plda_gguf_path;
    std::filesystem::path xvec_transform_gguf_path;
    bool enable_plda_refinement = true;
    double plda_sim_threshold = 0.55;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--words-json" && i + 1 < argc) {
            words_json_path = argv[++i];
            continue;
        }
        if (arg == "--diarization-json" && i + 1 < argc) {
            diarization_json_path = argv[++i];
            continue;
        }
        if (arg == "--out-prefix" && i + 1 < argc) {
            out_prefix = argv[++i];
            continue;
        }
        if (arg == "--max-gap-sec" && i + 1 < argc) {
            max_gap_sec = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--max-words" && i + 1 < argc) {
            max_words = std::max(0, std::stoi(argv[++i]));
            continue;
        }
        if (arg == "--max-duration-sec" && i + 1 < argc) {
            max_duration_sec = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--plda-gguf" && i + 1 < argc) {
            plda_gguf_path = argv[++i];
            continue;
        }
        if (arg == "--xvec-transform-gguf" && i + 1 < argc) {
            xvec_transform_gguf_path = argv[++i];
            continue;
        }
        if (arg == "--plda-sim-threshold" && i + 1 < argc) {
            plda_sim_threshold = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--disable-plda-refinement") {
            enable_plda_refinement = false;
            continue;
        }
        if (arg == "--prefer-exclusive") {
            prefer_exclusive = true;
            continue;
        }
        if (arg == "--prefer-regular") {
            prefer_exclusive = false;
            continue;
        }
        if (arg == "--split-on-hard-break") {
            split_on_hard_break = true;
            continue;
        }
        if (arg == "--no-split-on-hard-break") {
            split_on_hard_break = false;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        std::cerr << "unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 2;
    }

    if (words_json_path.empty() || diarization_json_path.empty() || out_prefix.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const json words_root = read_json_file(words_json_path);
        const json diar_root = read_json_file(diarization_json_path);

        std::vector<word_item> words = parse_words(words_root);
        const auto [diar_segments, diar_kind] = parse_diarization_segments(diar_root, prefer_exclusive);
        const std::vector<speaker_embedding> speaker_embeddings = parse_speaker_embeddings(diar_root);
        const plda_refinement_state plda_state = build_plda_refinement_state(
            plda_gguf_path,
            xvec_transform_gguf_path,
            plda_sim_threshold,
            speaker_embeddings,
            enable_plda_refinement);

        std::vector<word_item> speaker_words = assign_speakers_to_words(words, diar_segments, plda_state);
        std::vector<speaker_segment> speaker_segments = build_speaker_segments(
            speaker_words, max_gap_sec, max_words, max_duration_sec, split_on_hard_break);

        const std::filesystem::path out_words_tsv = out_prefix.string() + ".speaker_words.tsv";
        const std::filesystem::path out_speaker_txt = out_prefix.string() + ".speaker_transcript.txt";
        const std::filesystem::path out_speaker_srt = out_prefix.string() + ".speaker.srt";
        const std::filesystem::path out_speaker_json = out_prefix.string() + ".speaker_alignment.json";

        std::ostringstream words_tsv;
        words_tsv << "start_sec\tend_sec\tspeaker\tword\n";
        for (const auto & w : speaker_words) {
            words_tsv << fmt3(round3(w.start_sec)) << "\t"
                      << fmt3(round3(w.end_sec)) << "\t"
                      << sanitize_tsv(w.speaker) << "\t"
                      << sanitize_tsv(w.word) << "\n";
        }

        write_text_file(out_words_tsv, words_tsv.str());
        write_text_file(out_speaker_txt, segments_to_speaker_text(speaker_segments));
        write_text_file(out_speaker_srt, segments_to_srt(speaker_segments));

        std::map<std::string, int32_t> speaker_word_counts;
        for (const auto & w : speaker_words) {
            speaker_word_counts[w.speaker] += 1;
        }

        json out = json::object();
        out["source"] = {
            {"words_json", std::filesystem::absolute(words_json_path).string()},
            {"diarization_json", std::filesystem::absolute(diarization_json_path).string()},
        };
        out["config"] = {
            {"max_gap_sec", max_gap_sec},
            {"max_words", max_words},
            {"max_duration_sec", max_duration_sec},
            {"split_on_hard_break", split_on_hard_break},
            {"prefer_exclusive", prefer_exclusive},
            {"selected_diarization_kind", diar_kind},
            {"plda_refinement_enabled", enable_plda_refinement},
            {"plda_sim_threshold", plda_sim_threshold},
            {"plda_gguf", plda_gguf_path.empty() ? "" : std::filesystem::absolute(plda_gguf_path).string()},
            {"xvec_transform_gguf", xvec_transform_gguf_path.empty() ? "" : std::filesystem::absolute(xvec_transform_gguf_path).string()},
        };
        out["num_diarization_segments"] = static_cast<int64_t>(diar_segments.size());
        out["speaker_word_count"] = static_cast<int64_t>(speaker_words.size());
        out["speaker_segment_count"] = static_cast<int64_t>(speaker_segments.size());
        out["speaker_word_counts"] = speaker_word_counts;
        out["plda_refinement"] = {
            {"enabled", plda_state.enabled},
            {"status", plda_state.status},
            {"models_loaded", plda_state.models_loaded},
            {"speaker_embeddings_used", static_cast<int64_t>(speaker_embeddings.size())},
        };
        if (plda_state.models_loaded) {
            json sim = json::array();
            for (const auto & row : plda_state.pair_similarity) {
                for (const auto & col : row.second) {
                    if (row.first >= col.first) {
                        continue;
                    }
                    sim.push_back({
                        {"speaker_a", row.first},
                        {"speaker_b", col.first},
                        {"similarity", round3(col.second)},
                    });
                }
            }
            out["plda_refinement"]["pair_similarity"] = std::move(sim);
        }

        json j_words = json::array();
        for (const auto & w : speaker_words) {
            j_words.push_back({
                {"word", w.word},
                {"start_sec", round3(w.start_sec)},
                {"end_sec", round3(w.end_sec)},
                {"start_hms", format_hms(w.start_sec)},
                {"end_hms", format_hms(w.end_sec)},
                {"speaker", w.speaker},
                {"start_token_index", w.start_token_index},
                {"end_token_index", w.end_token_index},
            });
        }
        out["speaker_words"] = std::move(j_words);

        json j_segments = json::array();
        for (const auto & seg : speaker_segments) {
            j_segments.push_back({
                {"speaker", seg.speaker},
                {"start_sec", round3(seg.start_sec)},
                {"end_sec", round3(seg.end_sec)},
                {"start_hms", format_hms(seg.start_sec)},
                {"end_hms", format_hms(seg.end_sec)},
                {"num_words", seg.num_words},
                {"text", seg.text},
            });
        }
        out["speaker_segments"] = std::move(j_segments);

        write_text_file(out_speaker_json, out.dump(2));

        std::cout << "Speaker words TSV: " << out_words_tsv.string() << "\n";
        std::cout << "Speaker transcript TXT: " << out_speaker_txt.string() << "\n";
        std::cout << "Speaker subtitles SRT: " << out_speaker_srt.string() << "\n";
        std::cout << "Speaker alignment JSON: " << out_speaker_json.string() << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

#ifndef LLAMA_PYANNOTE_NO_MAIN
int main(int argc, char ** argv) {
    return llama_pyannote_align_main(argc, argv);
}
#endif

