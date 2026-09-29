import {
  useEffect,
  useRef,
  useState,
} from "react";

import { createTrainingExplanation } from "../../api/training";
import { C } from "../../config/theme";


const DEMO_REQUEST = {
  tenant_id: "bank-a",
  course_id: "gdpr-foundations",
  question_id: "gdpr-art5-data-minimisation-001",

  question:
    "Which GDPR principle requires personal data to be adequate, relevant and limited to what is necessary for the stated purpose?",

  options: [
    {
      option_id: "A",
      text: "Data minimisation",
    },
    {
      option_id: "B",
      text:
        "Collect as much personal data as might become useful later",
    },
  ],

  correct_option_ids: ["A"],
  selected_option_ids: ["B"],

  source_references: [
    "Regulation (EU) 2016/679, Article 5(1)(c)",
  ],

  reference_analysis:
    "Article 5(1)(c) requires personal data to be adequate, relevant and limited to what is necessary for the purposes for which it is processed.",

  jurisdiction: "LU",
  language: "en",
  top_k: 3,
};


const COPY = {
  en: {
    title: "GDPR Training Explanation Lab",
    subtitle:
      "Inspect how a learner answer becomes a grounded explanation.",
    run: "Run Article 5 demo",
    running: "Running…",
    waiting: "Waiting",
    processing: "Server processing",
    complete: "Complete",
    skipped: "Not enabled",
    notEvaluated: "Not evaluated",
    failed: "Failed",
    finalResult: "Final response",
    learnerResult: "Learner result",
    evidenceScore: "Evidence relevance",
    evidenceCount: "Evidence count",
    optionAnalysis: "Option explanations",
    evidence: "Evidence",
    rawResponse: "Raw API response",
    stages: [
      ["Request", "Question, options, answer key and learner selection"],
      ["Access Control", "Tenant, course and approved-source filtering"],
      ["Query Planning", "Question-level and option-level retrieval queries"],
      ["Retrieval", "Hybrid vector and keyword candidate retrieval"],
      ["Evidence Selection", "Coverage-aware evidence chosen for each option"],
      ["Generation", "Evidence-constrained explanation generation"],
      ["Grounding", "Citation and factual-support validation"],
      ["Final Response", "Learner-facing explanation and review status"],
    ],
  },

  zh: {
    title: "GDPR 训练解释实验室",
    subtitle: "观察学习者答案如何逐步转化为有依据的解释。",
    run: "运行 Article 5 示例",
    running: "运行中…",
    waiting: "等待执行",
    processing: "服务端处理中",
    complete: "完成",
    skipped: "尚未启用",
    notEvaluated: "尚未评估",
    failed: "失败",
    finalResult: "最终响应",
    learnerResult: "学习者结果",
    evidenceScore: "证据相关性",
    evidenceCount: "证据数量",
    optionAnalysis: "选项解释",
    evidence: "引用证据",
    rawResponse: "原始 API 响应",
    stages: [
      ["请求输入", "题目、选项、标准答案和学习者选择"],
      ["访问控制", "租户、课程和获准知识源过滤"],
      ["查询规划", "生成题目级和选项级检索查询"],
      ["混合检索", "向量与关键词候选召回"],
      ["证据选择", "为每个选项选择覆盖充分的证据"],
      ["解释生成", "在证据约束下生成学习解释"],
      ["依据验证", "检查引用合法性和事实支持"],
      ["最终响应", "学习者解释与审核状态"],
    ],
  },
};


function StageBadge({ label, color }) {
  return (
    <span
      style={{
        color,
        background: `${color}12`,
        border: `1px solid ${color}35`,
        borderRadius: 999,
        padding: "3px 8px",
        fontSize: 10,
        fontWeight: 700,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}


function Metric({ label, value }) {
  return (
    <div
      style={{
        background: C.bg,
        border: `1px solid ${C.border}`,
        borderRadius: 8,
        padding: 12,
      }}
    >
      <div
        style={{
          color: C.textMid,
          fontSize: 10,
          fontWeight: 700,
          marginBottom: 5,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        {label}
      </div>

      <div
        style={{
          color: C.text,
          fontSize: 14,
          fontWeight: 800,
        }}
      >
        {value}
      </div>
    </div>
  );
}


export function TrainingLabTab({ lang }) {
  const copy = COPY[lang] ?? COPY.en;

  const [runState, setRunState] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const controllerRef = useRef(null);


  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);


  const runDemo = async () => {
    controllerRef.current?.abort();

    const controller = new AbortController();
    controllerRef.current = controller;

    setRunState("running");
    setResult(null);
    setError(null);

    try {
      const response = await createTrainingExplanation(
        {
          ...DEMO_REQUEST,
          language: lang,
        },
        {
          signal: controller.signal,
        },
      );

      if (controllerRef.current !== controller) {
        return;
      }

      setResult(response);
      setRunState("complete");
    } catch (requestError) {
      if (requestError.name === "AbortError") {
        return;
      }

      setError(requestError);
      setRunState("failed");
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
    }
  };


  const getStageState = (index) => {
    if (runState === "idle") {
      return {
        label: copy.waiting,
        color: C.textDim,
      };
    }

    if (runState === "running") {
      if (index === 0) {
        return {
          label: copy.complete,
          color: C.green,
        };
      }

      return {
        label: copy.processing,
        color: C.orange,
      };
    }

    if (runState === "failed") {
      if (index === 7) {
        return {
          label: copy.failed,
          color: C.red,
        };
      }

      return {
        label: copy.notEvaluated,
        color: C.textDim,
      };
    }

    if (index <= 4 || index === 7) {
      return {
        label: copy.complete,
        color: C.green,
      };
    }

    if (index === 5 && !result?.generator_model) {
      return {
        label: copy.skipped,
        color: C.orange,
      };
    }

    if (index === 6 && result?.grounding_score == null) {
      return {
        label: copy.notEvaluated,
        color: C.orange,
      };
    }

    return {
      label: copy.complete,
      color: C.green,
    };
  };


  return (
    <section
      style={{
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        padding: 20,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 14,
          marginBottom: 20,
        }}
      >
        <div>
          <div
            style={{
              color: C.purple,
              fontSize: 11,
              fontWeight: 800,
              letterSpacing: "0.08em",
              marginBottom: 6,
            }}
          >
            DEVELOPMENT PIPELINE INSPECTOR
          </div>

          <h2
            style={{
              margin: 0,
              color: C.text,
              fontSize: 20,
            }}
          >
            {copy.title}
          </h2>

          <p
            style={{
              margin: "6px 0 0",
              color: C.textMid,
              fontSize: 13,
            }}
          >
            {copy.subtitle}
          </p>
        </div>

        <button
          type="button"
          onClick={runDemo}
          disabled={runState === "running"}
          style={{
            padding: "9px 15px",
            borderRadius: 8,
            border: "none",
            background:
              runState === "running"
                ? C.borderBright
                : C.purple,
            color: "#fff",
            cursor:
              runState === "running"
                ? "not-allowed"
                : "pointer",
            fontFamily: "inherit",
            fontSize: 12,
            fontWeight: 800,
          }}
        >
          {runState === "running"
            ? copy.running
            : copy.run}
        </button>
      </div>

      {error && (
        <div
          style={{
            marginBottom: 16,
            padding: 12,
            borderRadius: 8,
            background: `${C.red}10`,
            border: `1px solid ${C.red}35`,
            color: C.red,
            fontSize: 12,
          }}
        >
          HTTP {error.status ?? "error"}: {error.message}
        </div>
      )}

      <div
        style={{
          display: "grid",
          gap: 10,
        }}
      >
        {copy.stages.map(([title, description], index) => {
          const stageState = getStageState(index);

          return (
            <div
              key={title}
              style={{
                display: "grid",
                gridTemplateColumns:
                  "36px minmax(0, 1fr) auto",
                alignItems: "center",
                gap: 12,
                padding: 14,
                background: C.bg,
                border: `1px solid ${C.border}`,
                borderRadius: 10,
              }}
            >
              <div
                style={{
                  width: 30,
                  height: 30,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: "50%",
                  background: `${stageState.color}14`,
                  color: stageState.color,
                  fontSize: 12,
                  fontWeight: 800,
                }}
              >
                {index + 1}
              </div>

              <div>
                <div
                  style={{
                    color: C.text,
                    fontSize: 13,
                    fontWeight: 700,
                  }}
                >
                  {title}
                </div>

                <div
                  style={{
                    color: C.textMid,
                    fontSize: 12,
                    marginTop: 3,
                  }}
                >
                  {description}
                </div>
              </div>

              <StageBadge
                label={stageState.label}
                color={stageState.color}
              />
            </div>
          );
        })}
      </div>

      {result && (
        <div
          style={{
            marginTop: 20,
            paddingTop: 20,
            borderTop: `1px solid ${C.border}`,
          }}
        >
          <h3
            style={{
              margin: "0 0 12px",
              color: C.text,
              fontSize: 16,
            }}
          >
            {copy.finalResult}
          </h3>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(150px, 1fr))",
              gap: 10,
              marginBottom: 16,
            }}
          >
            <Metric
              label="Status"
              value={result.status}
            />

            <Metric
              label={copy.learnerResult}
              value={result.learner_result}
            />

            <Metric
              label={copy.evidenceScore}
              value={result.evidence_relevance_score}
            />

            <Metric
              label={copy.evidenceCount}
              value={result.evidence.length}
            />
          </div>

          <h4
            style={{
              margin: "0 0 8px",
              color: C.text,
              fontSize: 13,
            }}
          >
            {copy.optionAnalysis}
          </h4>

          <div
            style={{
              display: "grid",
              gap: 8,
              marginBottom: 16,
            }}
          >
            {result.option_explanations.map((option) => (
              <div
                key={option.option_id}
                style={{
                  padding: 12,
                  background: C.bg,
                  border: `1px solid ${C.border}`,
                  borderRadius: 8,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    gap: 8,
                    alignItems: "center",
                    marginBottom: 6,
                  }}
                >
                  <strong>
                    {option.option_id}
                  </strong>

                  <StageBadge
                    label={
                      option.is_correct
                        ? "Correct"
                        : "Incorrect"
                    }
                    color={
                      option.is_correct
                        ? C.green
                        : C.red
                    }
                  />

                  {option.selected_by_learner && (
                    <StageBadge
                      label="Selected"
                      color={C.orange}
                    />
                  )}
                </div>

                <div
                  style={{
                    color: C.textMid,
                    fontSize: 12,
                    lineHeight: 1.6,
                  }}
                >
                  {option.explanation}
                </div>

                <div
                  style={{
                    marginTop: 6,
                    color: C.purple,
                    fontSize: 11,
                    fontWeight: 700,
                  }}
                >
                  {option.evidence_ids.join(", ")}
                </div>
              </div>
            ))}
          </div>

          <h4
            style={{
              margin: "0 0 8px",
              color: C.text,
              fontSize: 13,
            }}
          >
            {copy.evidence}
          </h4>

          <div
            style={{
              display: "grid",
              gap: 8,
            }}
          >
            {result.evidence.map((evidence) => (
              <div
                key={evidence.evidence_id}
                style={{
                  padding: 12,
                  background: C.bg,
                  border: `1px solid ${C.border}`,
                  borderRadius: 8,
                }}
              >
                <div
                  style={{
                    color: C.text,
                    fontSize: 12,
                    fontWeight: 800,
                    marginBottom: 5,
                  }}
                >
                  {evidence.evidence_id}
                  {" · "}
                  {evidence.section}
                  {" · "}
                  {evidence.source}
                </div>

                <div
                  style={{
                    color: C.textMid,
                    fontSize: 12,
                    lineHeight: 1.6,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {evidence.excerpt}
                </div>
              </div>
            ))}
          </div>

          <details
            style={{
              marginTop: 16,
              color: C.textMid,
              fontSize: 12,
            }}
          >
            <summary
              style={{
                cursor: "pointer",
                fontWeight: 700,
              }}
            >
              {copy.rawResponse}
            </summary>

            <pre
              style={{
                overflow: "auto",
                padding: 12,
                background: C.bg,
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                fontSize: 11,
              }}
            >
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </section>
  );
}