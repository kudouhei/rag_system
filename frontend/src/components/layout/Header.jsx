import { C } from "../../config/theme";
import { I18N } from "../../i18n/index.jsx";
import { Tag } from "../ui.jsx";


const WORKSPACES = [
  {
    key: "rag",
    labels: {
      en: "RAG Explorer",
      zh: "RAG 检索",
    },
  },
  {
    key: "training",
    labels: {
      en: "GDPR Training",
      zh: "GDPR 训练",
    },
  },
];


export const Header = ({
  lang,
  setLang,
  setQuery,
  agentMode,
  workspace,
  setWorkspace,
  t,
}) => {
  const isTraining = workspace === "training";

  const title = isTraining
    ? "GDPR Learning & Explanation"
    : "Regulatory Document Intelligence RAG";

  const subtitle = isTraining
    ? (
      lang === "zh"
        ? "选项级证据检索 · 学习解释 · 依据验证"
        : "Option-level evidence · Learning explanations · Grounding"
    )
    : t("appSubtitle");

  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: 14,
        marginBottom: 24,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <div
          style={{
            width: 38,
            height: 38,
            borderRadius: 10,
            background: `linear-gradient(
              135deg,
              ${C.accent}28,
              ${C.purple}28
            )`,
            border: `1px solid ${C.accent}44`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 20,
          }}
        >
          {isTraining ? "◎" : "⚡"}
        </div>

        <div>
          <h1
            style={{
              margin: 0,
              fontSize: 18,
              fontWeight: 800,
              color: C.text,
            }}
          >
            {title}
          </h1>

          <p
            style={{
              margin: 0,
              fontSize: 11,
              color: C.textMid,
            }}
          >
            {subtitle}
          </p>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 10,
        }}
      >
        <nav
          aria-label="Workspace"
          style={{
            display: "flex",
            borderRadius: 8,
            overflow: "hidden",
            border: `1px solid ${C.borderBright}`,
          }}
        >
          {WORKSPACES.map((item) => {
            const active = workspace === item.key;

            return (
              <button
                key={item.key}
                type="button"
                onClick={() => {
                  setWorkspace(item.key);
                }}
                style={{
                  padding: "7px 12px",
                  cursor: "pointer",
                  border: "none",
                  fontFamily: "inherit",
                  fontSize: 12,
                  fontWeight: 700,
                  background: active
                    ? `${C.purple}18`
                    : "transparent",
                  color: active
                    ? C.purple
                    : C.textMid,
                }}
              >
                {item.labels[lang] ?? item.labels.en}
              </button>
            );
          })}
        </nav>

        <div
          style={{
            display: "flex",
            borderRadius: 8,
            overflow: "hidden",
            border: `1px solid ${C.borderBright}`,
            fontSize: 12,
            fontWeight: 700,
          }}
        >
          {["en", "zh"].map((language) => (
            <button
              key={language}
              type="button"
              onClick={() => {
                setLang(language);
                setQuery(
                  I18N[language].sampleQueries[0],
                );
              }}
              style={{
                padding: "5px 14px",
                cursor: "pointer",
                fontFamily: "inherit",
                fontWeight: 700,
                fontSize: 12,
                letterSpacing: "0.04em",
                background:
                  lang === language
                    ? C.accent
                    : "transparent",
                color:
                  lang === language
                    ? "#fff"
                    : C.textMid,
                border: "none",
              }}
            >
              {language === "zh" ? "中文" : "EN"}
            </button>
          ))}
        </div>

        <div
          style={{
            display: "flex",
            gap: 6,
            flexWrap: "wrap",
          }}
        >
          {agentMode && (
            <Tag
              label={t("badge_agent")}
              color={C.orange}
            />
          )}
        </div>
      </div>
    </header>
  );
};