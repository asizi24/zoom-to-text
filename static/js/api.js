// Fetch wrapper: parses the server's error envelope ({detail, code,
// request_id}) into a typed ApiError and turns an expired session into a
// redirect to /login instead of a dead UI.

export class ApiError extends Error {
  constructor(message, { status = 0, code = '', requestId = '' } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

// Low-level: returns the Response on 2xx, throws ApiError otherwise.
// Streaming callers (SSE-over-fetch chat) use this and read response.body.
export async function apiFetch(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError('שגיאת רשת — בדוק את החיבור לשרת', { code: 'network' });
  }
  if (response.status === 401) {
    // Session expired — the only recovery is a fresh magic-link login.
    window.location.href = '/login';
    throw new ApiError('נדרשת התחברות מחדש', { status: 401, code: 'unauthorized' });
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || `שגיאה בשרת (HTTP ${response.status})`, {
      status: response.status,
      code: body.code || '',
      requestId: body.request_id || '',
    });
  }
  return response;
}

// High-level: JSON in → JSON out.
export async function apiJson(path, options = {}) {
  const response = await apiFetch(path, options);
  if (response.status === 204) return null;
  return response.json();
}

export function postJson(path, body) {
  return apiJson(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}
