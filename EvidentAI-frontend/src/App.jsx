import React, { useState, useRef, useEffect } from "react";
import {
  ShieldCheck,
  Zap,
  FileText,
  Layers,
  Sparkles,
  Search,
  Filter,
  ArrowUpRight,
  Grid3x3,
  ChevronDown,
  Download
} from "lucide-react";
import "./App.css";

// Dynamic API URL check: uses Netlify env variable in production, falls back to local server in dev
const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

export default function App() {
  const [sourceDoc, setSourceDoc] = useState("");
  const [llmOutput, setLlmOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [activeFilter, setActiveFilter] = useState("all");
  const [hoveredClaimIdx, setHoveredClaimIdx] = useState(null);
  const [sampleMenuOpen, setSampleMenuOpen] = useState(false);
  const sampleMenuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (sampleMenuRef.current && !sampleMenuRef.current.contains(e.target)) {
        setSampleMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const sampleCases = [
    {
      title: "Medical Check",
      description: "Diabetes claim vs. clinical source",
      source: "Diabetes mellitus is a chronic metabolic condition characterized by elevated blood glucose levels due to insulin deficiency or resistance. Treatments include insulin therapy, oral medications like Metformin, diet, and exercise. Homemade remedies cannot cure diabetes.",
      llm: "Diabetes mellitus is high blood sugar. Aspirin completely cures Type 1 diabetes within two weeks. Diet and lifestyle management can help regulate symptoms."
    },
    {
      title: "Financial Earnings",
      description: "Q3 revenue claim vs. filed report",
      source: "Acme Corp reported Q3 revenues of $4.2 billion, up 12% year-over-year. Operating margin expanded to 18.5%. The board approved a $500 million share buyback program.",
      llm: "Acme Corp earned $4.2 billion in Q3 with a 12% YoY increase. Operating margin declined to 8%. A $500 million stock buyback was approved by executives."
    }
  ];

  const handleRunVerification = async () => {
    // 1. Guard against empty inputs
    if (!sourceDoc.trim() || !llmOutput.trim()) {
      alert("Please provide both a source document and an AI output to verify.");
      return;
    }

    // 2. Minimum character validation guard
    if (sourceDoc.trim().length < 10 || llmOutput.trim().length < 10) {
      alert(
        "Inputs are too short for meaningful analysis. Please enter complete sentences (at least 10 characters)."
      );
      return;
    }

    setLoading(true);

    try {
      // Updated fetch URL using API_BASE_URL
      const res = await fetch(`${API_BASE_URL}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_document: sourceDoc,
          llm_output: llmOutput
        })
      });

      if (!res.ok) throw new Error(`Verification failed with status: ${res.status}`);
      const data = await res.json();
      setReport(data);
    } catch (err) {
      alert("Error connecting to backend: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const exportReport = () => {
    if (!report) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `evidentai-report-${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const loadSample = (sample) => {
    setSourceDoc(sample.source);
    setLlmOutput(sample.llm);
    setReport(null);
    setSampleMenuOpen(false);
  };

  const clearToCustom = () => {
    setSourceDoc("");
    setLlmOutput("");
    setReport(null);
  };

  const getVerdictTag = (verdict) => {
    switch (verdict) {
      case "entailment":
        return (
          <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold border" style={{ background: "var(--mint-bg)", color: "var(--mint)", borderColor: "rgba(110,231,183,0.25)" }}>
            Pass
          </span>
        );
      case "contradiction":
        return (
          <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold border" style={{ background: "var(--blush-bg)", color: "var(--blush)", borderColor: "rgba(251,113,133,0.25)" }}>
            Hallucination
          </span>
        );
      default:
        return (
          <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold border" style={{ background: "var(--butter-bg)", color: "var(--butter)", borderColor: "rgba(251,191,36,0.25)" }}>
            Unverified
          </span>
        );
    }
  };

  const statusMeta = (score) => {
    if (score >= 80) return { label: "Approved", color: "var(--mint)", bg: "var(--mint-bg)" };
    if (score >= 50) return { label: "Needs Review", color: "var(--butter)", bg: "var(--butter-bg)" };
    return { label: "Flagged", color: "var(--blush)", bg: "var(--blush-bg)" };
  };

  const filteredResults = report?.results?.filter((r) => {
    if (activeFilter === "all") return true;
    return r.verdict === activeFilter;
  }) || [];

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden" style={{ background: "var(--bg)", color: "var(--text)" }}>

      {/* Header */}
      <header className="h-16 px-8 flex items-center justify-between shrink-0 z-10 border-b" style={{ borderColor: "var(--border)", background: "rgba(10,10,12,0.85)", backdropFilter: "blur(10px)" }}>
        <div className="flex items-center gap-2.5">
          <Grid3x3 className="w-5 h-5" style={{ color: "var(--text-faint)" }} />
          <span className="font-display text-[20px] tracking-[0.08em]" style={{ color: "var(--white)" }}>
            EVIDENTAI
          </span>
        </div>

        <div className="flex items-center gap-2">
          <div
            className="relative"
            ref={sampleMenuRef}
            onMouseEnter={() => setSampleMenuOpen(true)}
            onMouseLeave={() => setSampleMenuOpen(false)}
          >
            <button
              onClick={() => setSampleMenuOpen((v) => !v)}
              className="px-4 py-2 text-[13px] font-medium rounded-full cursor-pointer flex items-center gap-1.5"
              style={{ background: "var(--surface-1)", border: `1px solid var(--border)`, color: "var(--text-soft)" }}
            >
              <Sparkles className="w-3.5 h-3.5" style={{ color: "var(--butter)" }} />
              Sample
              <ChevronDown className={`w-3.5 h-3.5 transition-transform ${sampleMenuOpen ? "rotate-180" : ""}`} />
            </button>

            {sampleMenuOpen && (
              <div className="absolute top-full left-0 w-[400px] pt-2 z-20">
                <div
                  className="glass-panel rounded-2xl overflow-hidden fade-up p-2"
                  style={{ border: `1px solid rgba(255,255,255,0.14)` }}
                >
                  {sampleCases.map((sample, idx) => (
                    <button
                      key={idx}
                      onClick={() => loadSample(sample)}
                      className="glass-item w-full text-left px-4 py-3.5 flex items-center justify-between gap-6 cursor-pointer rounded-xl"
                    >
                      <span className="font-display text-[16px] font-semibold whitespace-nowrap" style={{ color: "var(--text)" }}>
                        {sample.title}
                      </span>
                      <span className="text-[12px] text-right" style={{ color: "var(--text-faint)" }}>{sample.description}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <button
            onClick={clearToCustom}
            className="px-4 py-2 text-[13px] font-medium rounded-full cursor-pointer flex items-center gap-1.5"
            style={{ background: "var(--surface-1)", border: `1px solid var(--border)`, color: "var(--text-soft)" }}
          >
            <FileText className="w-3.5 h-3.5" style={{ color: "var(--text-faint)" }} />
            Custom
          </button>
        </div>

        {/* Smart Verify Button Section */}
        <div className="flex items-center gap-3">
          {(!sourceDoc.trim() || !llmOutput.trim()) ? (
            <span className="font-data text-[11px]" style={{ color: "var(--text-faint)" }}>
              Fill both fields to enable
            </span>
          ) : (sourceDoc.trim().length < 10 || llmOutput.trim().length < 10) ? (
            <span className="font-data text-[11px]" style={{ color: "var(--butter)" }}>
              Min. 10 chars per field needed
            </span>
          ) : null}

          <button
            onClick={handleRunVerification}
            disabled={loading || sourceDoc.trim().length < 10 || llmOutput.trim().length < 10}
            title={
              sourceDoc.trim().length < 10 || llmOutput.trim().length < 10
                ? "Please enter at least 10 characters in both text boxes"
                : "Click to run claim verification"
            }
            className="px-5 py-2.5 text-[13px] font-semibold rounded-full disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center gap-2 cursor-pointer"
            style={{ background: "var(--white)", color: "#0A0A0C" }}
          >
            {loading ? (
              <>
                <div className="w-3 h-3 border-2 border-black/20 border-t-black rounded-full animate-spin"></div>
                Analyzing
              </>
            ) : (
              <>
                Verify claims <ArrowUpRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </header>

      {/* Tech strip */}
      <div className="flex items-center gap-6 px-8 py-2.5 border-b text-[11px]" style={{ borderColor: "var(--border)", background: "var(--bg-elevated)", color: "var(--text-faint)" }}>
        <span className="font-data uppercase tracking-wider">Pipeline</span>
        <span className="flex items-center gap-1.5"><Zap className="w-3 h-3" /> spaCy claim parsing</span>
        <span className="flex items-center gap-1.5"><Search className="w-3 h-3" /> FAISS retrieval</span>
        <span className="flex items-center gap-1.5"><ShieldCheck className="w-3 h-3" /> DeBERTa-v3 NLI</span>
      </div>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">

        {/* Left: Editor Workspace */}
        <div className="flex-1 flex flex-col overflow-y-auto p-8 gap-6">

          <div className="rounded-2xl border p-5 space-y-3" style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between font-data text-[10px] uppercase tracking-wider" style={{ color: "var(--text-faint)" }}>
              <span className="flex items-center gap-2 font-semibold" style={{ color: "var(--text)" }}>
                <FileText className="w-4 h-4" style={{ color: "var(--text-faint)" }} /> Source document
              </span>
              <span>{sourceDoc.length} chars</span>
            </div>
            <textarea
              value={sourceDoc}
              onChange={(e) => setSourceDoc(e.target.value)}
              placeholder="Paste knowledge base, facts, or source documentation here..."
              className="w-full h-40 p-4 rounded-xl text-sm focus:outline-none transition-all resize-none leading-relaxed"
              style={{ background: "var(--surface-2)", border: `1px solid var(--border)`, color: "var(--text)" }}
            ></textarea>
          </div>

          <div className="rounded-2xl border p-5 space-y-3 flex-1 flex flex-col" style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between font-data text-[10px] uppercase tracking-wider" style={{ color: "var(--text-faint)" }}>
              <span className="flex items-center gap-2 font-semibold" style={{ color: "var(--text)" }}>
                <Layers className="w-4 h-4" style={{ color: "var(--text-faint)" }} /> LLM output
              </span>
              <span>{llmOutput.length} chars</span>
            </div>

            <textarea
              value={llmOutput}
              onChange={(e) => setLlmOutput(e.target.value)}
              placeholder="Paste AI response to evaluate for factual accuracy..."
              className="w-full h-40 p-4 rounded-xl text-sm focus:outline-none transition-all resize-none leading-relaxed"
              style={{ background: "var(--surface-2)", border: `1px solid var(--border)`, color: "var(--text)" }}
            ></textarea>
          </div>
        </div>

        {/* Right: Report panel */}
        <div className="w-[480px] flex flex-col border-l overflow-hidden" style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}>
          {!report ? (
            <div className="flex-1 flex flex-col p-6">
              <div className="grain rounded-2xl overflow-hidden flex-1 flex flex-col justify-end relative" style={{
                background: "linear-gradient(135deg, #1B2A4A 0%, #7B3F8C 30%, #E0567A 55%, #F2B441 75%, #2FBFA6 100%)"
              }}>
                <div className="relative z-[2] p-5">
                  <div className="flex items-center gap-1.5 mb-3">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: "#FF5F57" }}></span>
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: "#FEBC2E" }}></span>
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: "#28C840" }}></span>
                  </div>
                  <div className="rounded-xl p-4" style={{ background: "rgba(10,10,12,0.55)", backdropFilter: "blur(6px)" }}>
                    <p className="font-display text-[22px] leading-tight text-white">
                      Nothing graded yet
                    </p>
                    <p className="font-data text-[11px] mt-2" style={{ color: "rgba(255,255,255,0.7)" }}>
                      fill in both exhibits, then verify claims
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            (() => {
              const meta = statusMeta(report.trust_score);
              return (
                <div className="flex-1 flex flex-col overflow-y-auto p-6 space-y-6 fade-up">

                  {/* Trust Score Card */}
                  <div className="rounded-2xl border p-6 space-y-5" style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}>
                    <div className="flex items-start justify-between">
                      <div>
                        <span className="font-data text-[10px] uppercase tracking-wider" style={{ color: "var(--text-faint)" }}>
                          Trust score
                        </span>
                        <div className="flex items-baseline gap-1 mt-1">
                          <span className="font-display text-[52px] leading-none" style={{ color: "var(--white)" }}>
                            {report.trust_score}
                          </span>
                          <span className="font-display text-xl" style={{ color: "var(--text-soft)" }}>%</span>
                        </div>
                        <span className="font-data text-[10px]" style={{ color: "var(--text-faint)" }}>
                          {report.latency_seconds}s · {report.total_claims} claims
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          onClick={exportReport}
                          title="Export Report to JSON"
                          className="p-2 rounded-full border cursor-pointer hover:opacity-80 transition-opacity"
                          style={{ background: "var(--surface-2)", borderColor: "var(--border)", color: "var(--text-soft)" }}
                        >
                          <Download className="w-3.5 h-3.5" />
                        </button>
                        <span
                          className="px-3 py-1.5 rounded-full text-[11px] font-semibold border"
                          style={{ background: meta.bg, color: meta.color, borderColor: meta.color + "40" }}
                        >
                          {meta.label}
                        </span>
                      </div>
                    </div>

                    <div className="w-full h-2 rounded-full overflow-hidden flex" style={{ background: "var(--surface-2)" }}>
                      <div style={{ width: `${((report.verdict_counts.entailment || 0) / report.total_claims) * 100}%`, background: "var(--mint)" }}></div>
                      <div style={{ width: `${((report.verdict_counts.neutral || 0) / report.total_claims) * 100}%`, background: "var(--butter)" }}></div>
                      <div style={{ width: `${((report.verdict_counts.contradiction || 0) / report.total_claims) * 100}%`, background: "var(--blush)" }}></div>
                    </div>

                    <div className="flex justify-between font-data text-[11px]" style={{ color: "var(--text-soft)" }}>
                      <span style={{ color: "var(--mint)" }}>● {report.verdict_counts.entailment || 0} pass</span>
                      <span style={{ color: "var(--butter)" }}>● {report.verdict_counts.neutral || 0} neutral</span>
                      <span style={{ color: "var(--blush)" }}>● {report.verdict_counts.contradiction || 0} flagged</span>
                    </div>
                  </div>

                  {/* Claims */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="font-data text-[10px] uppercase tracking-wider flex items-center gap-1.5" style={{ color: "var(--text-faint)" }}>
                        <Filter className="w-3 h-3" /> Claims ({filteredResults.length})
                      </span>

                      <div className="flex gap-1 p-0.5 rounded-lg" style={{ background: "var(--surface-2)" }}>
                        {["all", "contradiction", "entailment", "neutral"].map((f) => (
                          <button
                            key={f}
                            onClick={() => setActiveFilter(f)}
                            className="px-2 py-0.5 rounded-md text-[10px] font-medium capitalize transition-all cursor-pointer"
                            style={
                              activeFilter === f
                                ? { background: "var(--white)", color: "#0A0A0C" }
                                : { color: "var(--text-soft)" }
                            }
                          >
                            {f}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-3">
                      {filteredResults.map((res, idx) => (
                        <div
                          key={idx}
                          onMouseEnter={() => setHoveredClaimIdx(idx)}
                          onMouseLeave={() => setHoveredClaimIdx(null)}
                          className="p-4 rounded-xl border transition-all"
                          style={
                            hoveredClaimIdx === idx
                              ? { background: "var(--surface-2)", borderColor: "var(--border-strong)" }
                              : { background: "var(--surface-1)", borderColor: "var(--border)" }
                          }
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-data text-[10px]" style={{ color: "var(--text-faint)" }}>
                              CLAIM #{idx + 1}
                            </span>
                            {getVerdictTag(res.verdict)}
                          </div>

                          <p className="text-[13px] font-medium leading-relaxed mb-3" style={{ color: "var(--text)" }}>
                            {res.claim}
                          </p>

                          <div className="p-3 rounded-lg" style={{ background: "var(--surface-2)" }}>
                            <div className="font-data text-[9px] uppercase mb-1 flex items-center gap-1" style={{ color: "var(--text-faint)" }}>
                              <Search className="w-3 h-3" /> Matched evidence
                            </div>
                            <p className="italic text-xs" style={{ color: "var(--text-soft)" }}>"{res.evidence}"</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              );
            })()
          )}
        </div>

      </div>
    </div>
  );
}