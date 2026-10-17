// pybind11 module `stsupport._ccgcky`.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <memory>
#include <thread>

#include "category.hpp"
#include "grammar.hpp"
#include "parser.hpp"

namespace py = pybind11;
using namespace ccg;

namespace {

// Convert C++ exceptions into Python errors explicitly. The function must be
// called with the GIL held. (pybind11's automatic translator chain does not
// fire reliably with every MSVC toolset; setting the error ourselves and
// throwing error_already_set always works.)
template <typename F>
auto guarded(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const py::error_already_set&) {
        throw;
    } catch (const ParseCancelled& e) {
        PyErr_SetString(PyExc_InterruptedError, e.what());
        throw py::error_already_set();
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        throw py::error_already_set();
    } catch (const std::out_of_range& e) {
        PyErr_SetString(PyExc_IndexError, e.what());
        throw py::error_already_set();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}

RuleConfig config_from_dict(const py::dict& d) {
    RuleConfig cfg;
    auto get = [&](const char* key) -> py::object {
        if (d.contains(key)) return py::reinterpret_borrow<py::object>(d[key]);
        return py::none();
    };
    if (!get("max_composition_degree").is_none()) cfg.max_composition_degree = get("max_composition_degree").cast<int>();
    if (!get("harmonic_backward_composition").is_none()) cfg.harmonic_backward_composition = get("harmonic_backward_composition").cast<bool>();
    if (!get("crossed_composition").is_none()) cfg.crossed_composition = get("crossed_composition").cast<bool>();
    if (!get("bx_require_vp").is_none()) cfg.bx_require_vp = get("bx_require_vp").cast<bool>();
    if (!get("comma_as_conj").is_none()) cfg.comma_as_conj = get("comma_as_conj").cast<bool>();
    if (!get("type_raising").is_none()) cfg.type_raising = get("type_raising").cast<bool>();
    if (!get("type_changing").is_none()) cfg.type_changing = get("type_changing").cast<bool>();
    if (!get("max_unary_depth").is_none()) cfg.max_unary_depth = get("max_unary_depth").cast<int>();
    if (!get("use_seen_rules").is_none()) cfg.use_seen_rules = get("use_seen_rules").cast<bool>();
    if (!get("unary_penalty").is_none()) cfg.unary_penalty = get("unary_penalty").cast<double>();
    if (!get("type_raise_penalty").is_none()) cfg.type_raise_penalty = get("type_raise_penalty").cast<double>();
    if (!get("cell_beam").is_none()) cfg.cell_beam = get("cell_beam").cast<int>();
    if (!get("left_punct").is_none()) cfg.left_punct = get("left_punct").cast<std::vector<std::string>>();
    if (!get("right_punct").is_none()) cfg.right_punct = get("right_punct").cast<std::vector<std::string>>();
    if (!get("type_change_rules").is_none()) cfg.type_change_rules = get("type_change_rules").cast<std::vector<std::pair<std::string, std::string>>>();
    if (!get("type_raise_rules").is_none()) cfg.type_raise_rules = get("type_raise_rules").cast<std::vector<std::pair<std::string, std::string>>>();
    if (!get("root_categories").is_none()) cfg.root_categories = get("root_categories").cast<std::vector<std::string>>();
    if (!get("seen_rules").is_none()) cfg.seen_rules = get("seen_rules").cast<std::vector<std::pair<std::string, std::string>>>();
    return cfg;
}

py::dict output_to_py(const ParseOutput& out) {
    py::dict d;
    d["ok"] = out.ok;
    d["score"] = out.score;
    d["root_cat"] = out.root_cat;
    d["root"] = out.root;
    d["chosen"] = out.chosen;
    d["chart_items"] = out.chart_items;
    d["root_items"] = out.root_items;
    d["root_cell_cats"] = out.root_cell_cats;
    py::list nodes;
    for (const Node& n : out.nodes) {
        py::dict nd;
        nd["cat"] = n.cat;
        nd["rule"] = n.rule;
        nd["head"] = n.head;
        nd["span"] = py::make_tuple(n.start, n.end);
        nd["children"] = n.children;
        if (n.token >= 0) {
            nd["token"] = n.token;
            nd["cand"] = n.cand;
            nd["word"] = n.word;
            nd["supertag"] = n.supertag;
            nd["logp"] = n.logp;
        }
        nodes.append(nd);
    }
    d["nodes"] = nodes;
    return d;
}

class Engine {
public:
    explicit Engine(const py::dict& config) {
        guarded([&] {
            std::vector<std::string> strip = {"nb"};
            if (config.contains("strip_features")) strip = config["strip_features"].cast<std::vector<std::string>>();
            std::vector<std::string> punct = {",", ".", ":", ";", "LRB", "RRB", "LQU", "RQU"};
            if (config.contains("punct_atoms")) punct = config["punct_atoms"].cast<std::vector<std::string>>();
            store_ = std::make_unique<CategoryStore>(strip, punct);
            grammar_ = std::make_unique<Grammar>(*store_, config_from_dict(config));
            lexicon_ = std::make_unique<Lexicon>(*store_);
        });
    }

    py::dict parse(const std::vector<std::string>& tokens, const Lattice& lattice,
                   const std::shared_ptr<ParseCancellation>& cancellation) {
        return guarded([&] {
            ParseOutput out;
            {
                py::gil_scoped_release release;
                Parser parser(*store_, *grammar_, *lexicon_, cancellation.get());
                out = parser.parse(tokens, lattice);
            }
            return output_to_py(out);
        });
    }

    py::list parse_batch(const std::vector<std::vector<std::string>>& tokens,
                         const std::vector<Lattice>& lattices, int n_threads) {
        return guarded([&] {
            if (tokens.size() != lattices.size()) throw std::invalid_argument("tokens/lattices length mismatch");
            std::vector<ParseOutput> outs(tokens.size());
            {
                py::gil_scoped_release release;
                unsigned hw = std::thread::hardware_concurrency();
                int threads = n_threads > 0 ? n_threads : static_cast<int>(hw ? hw : 1);
                threads = std::max(1, std::min<int>(threads, static_cast<int>(tokens.size())));
                std::atomic<std::size_t> next{0};
                auto worker = [&]() {
                    Parser parser(*store_, *grammar_, *lexicon_);
                    while (true) {
                        std::size_t i = next.fetch_add(1);
                        if (i >= tokens.size()) break;
                        outs[i] = parser.parse(tokens[i], lattices[i]);
                    }
                };
                std::vector<std::thread> pool;
                for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
                worker();
                for (auto& th : pool) th.join();
            }
            py::list result;
            for (const ParseOutput& out : outs) result.append(output_to_py(out));
            return result;
        });
    }

    std::string normalize(const std::string& cat) {
        return guarded([&] { return store_->str(store_->parse(cat)); });
    }

    bool is_valid(const std::string& cat) {
        return guarded([&] { return lexicon_->lookup(cat).valid; });
    }

    py::list combine(const std::string& left, const std::string& right) {
        return guarded([&] {
            py::list res;
            CatId l = store_->parse(left), r = store_->parse(right);
            for (const RuleResult& rr : grammar_->binary(l, r))
                res.append(py::make_tuple(store_->str(rr.cat), std::string(rule_name(rr.rule)), static_cast<int>(rr.head)));
            return res;
        });
    }

    py::list unary(const std::string& cat) {
        return guarded([&] {
            py::list res;
            for (const RuleResult& rr : grammar_->unary(store_->parse(cat)))
                res.append(py::make_tuple(store_->str(rr.cat), std::string(rule_name(rr.rule))));
            return res;
        });
    }

    bool is_root(const std::string& cat) {
        return guarded([&] { return grammar_->is_root(store_->parse(cat)); });
    }

    std::size_t n_categories() const { return store_->size(); }

private:
    std::unique_ptr<CategoryStore> store_;
    std::unique_ptr<Grammar> grammar_;
    std::unique_ptr<Lexicon> lexicon_;
};

}  // namespace

PYBIND11_MODULE(_ccgcky, m) {
    m.doc() = "Viterbi CKY over CCG supertag lattices (Eisner normal form)";
    py::class_<ParseCancellation, std::shared_ptr<ParseCancellation>>(m, "ParseCancellation")
        .def(py::init<>())
        .def("cancel", &ParseCancellation::cancel);
    py::class_<Engine>(m, "Engine")
        .def(py::init<const py::dict&>(), py::arg("config"))
        .def("parse", &Engine::parse, py::arg("tokens"), py::arg("lattice"), py::arg("cancellation") = nullptr,
             "Parse one sentence; lattice[i] = [(category, logp), ...] for token i.")
        .def("parse_batch", &Engine::parse_batch, py::arg("tokens"), py::arg("lattices"), py::arg("n_threads") = 0,
             "Parse many sentences in parallel C++ threads.")
        .def("normalize", &Engine::normalize, py::arg("category"))
        .def("is_valid", &Engine::is_valid, py::arg("supertag"))
        .def("combine", &Engine::combine, py::arg("left"), py::arg("right"))
        .def("unary", &Engine::unary, py::arg("category"))
        .def("is_root", &Engine::is_root, py::arg("category"))
        .def_property_readonly("n_categories", &Engine::n_categories);
}
