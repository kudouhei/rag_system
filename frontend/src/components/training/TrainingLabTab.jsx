import { C } from "../../config/theme";

const COPY = {
  en: {
    title: "GDPR Training Explanation Lab",
    subtitle:
      "Inspect how a learner answer becomes a grounded explanation.",
    waiting: "Waiting",
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
    waiting: "等待执行",
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

export function TrainingLabTab({ lang }) {
  const copy = COPY[lang] ?? COPY.en;

  return (
    <section
      style={{
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        padding: 20,
      }}
    >
      <div style={{ marginBottom: 20 }}>
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

      <div
        style={{
          display: "grid",
          gap: 10,
        }}
      >
        {copy.stages.map(([title, description], index) => (
          <div
            key={title}
            style={{
              display: "grid",
              gridTemplateColumns: "36px minmax(0, 1fr) auto",
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
                background: `${C.accent}14`,
                color: C.accent,
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

            <span
              style={{
                color: C.textDim,
                background: C.surface,
                border: `1px solid ${C.border}`,
                borderRadius: 999,
                padding: "3px 8px",
                fontSize: 10,
                fontWeight: 700,
              }}
            >
              {copy.waiting}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}