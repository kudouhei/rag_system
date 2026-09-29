import { API_BASE } from "../config/api";


const TRAINING_EXPLANATION_ENDPOINT =
  `${API_BASE}/api/v1/training/explanations`;


async function readResponseBody(response) {
  const rawBody = await response.text();

  if (!rawBody) {
    return null;
  }

  try {
    return JSON.parse(rawBody);
  } catch {
    return {
      raw: rawBody,
    };
  }
}


export async function createTrainingExplanation(
  payload,
  { signal } = {},
) {
  const response = await fetch(
    TRAINING_EXPLANATION_ENDPOINT,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(payload),
      signal,
    },
  );

  const responseBody = await readResponseBody(response);

  if (!response.ok) {
    const detail = responseBody?.detail;

    const message =
      typeof detail === "string"
        ? detail
        : detail?.message
          || `Training API returned HTTP ${response.status}`;

    const error = new Error(message);

    error.status = response.status;
    error.responseBody = responseBody;

    throw error;
  }

  return responseBody;
}