import { API_BASE } from "../config/api";


const TRAINING_API_BASE =
  `${API_BASE}/api/v1/training`;

const TRAINING_EXPLANATION_ENDPOINT =
  `${TRAINING_API_BASE}/explanations`;

const TRAINING_DEBUG_ENDPOINT =
  `${TRAINING_API_BASE}/explanations/debug`;


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


async function postTrainingRequest(
  endpoint,
  payload,
  { signal } = {},
) {
  const response = await fetch(
    endpoint,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(payload),
      signal,
    },
  );

  const responseBody = await readResponseBody(
    response
  );

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


export function createTrainingExplanation(
  payload,
  options,
) {
  return postTrainingRequest(
    TRAINING_EXPLANATION_ENDPOINT,
    payload,
    options,
  );
}


export function debugTrainingExplanation(
  payload,
  options,
) {
  return postTrainingRequest(
    TRAINING_DEBUG_ENDPOINT,
    payload,
    options,
  );
}