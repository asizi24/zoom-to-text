// Reconnecting EventSource with exponential backoff + jitter.
//
// Native EventSource retries on its own schedule and gives up permanently on
// some error classes; this wrapper takes full control: every error closes the
// source and schedules a reconnect at min(1s·2^n, 30s) + jitter. A received
// message resets the failure counter. After maxFailures consecutive failures
// (no message in between) onPermanentFailure fires so the caller can degrade
// to polling — the caller may still create a fresh instance later to attempt
// an upgrade back to SSE.
export class ReconnectingEventSource {
  constructor(url, { onMessage, onPermanentFailure = null, maxFailures = 5, maxDelayMs = 30000 } = {}) {
    this.url = url;
    this.onMessage = onMessage;
    this.onPermanentFailure = onPermanentFailure;
    this.maxFailures = maxFailures;
    this.maxDelayMs = maxDelayMs;
    this._failures = 0;
    this._es = null;
    this._timer = null;
    this._closed = false;
  }

  start() {
    this._connect();
  }

  stop() {
    this._closed = true;
    clearTimeout(this._timer);
    this._timer = null;
    if (this._es) { this._es.close(); this._es = null; }
  }

  _connect() {
    if (this._closed) return;
    let es;
    try {
      es = new EventSource(this.url);
    } catch {
      this._onFailure();
      return;
    }
    this._es = es;

    es.onmessage = (e) => {
      this._failures = 0; // healthy stream — reset the backoff ladder
      this.onMessage(e);
    };
    es.onerror = () => {
      es.close();
      if (this._closed || this._es !== es) return;
      this._es = null;
      this._onFailure();
    };
  }

  _onFailure() {
    this._failures++;
    if (this._failures >= this.maxFailures) {
      this.onPermanentFailure?.();
      return;
    }
    const delay = Math.min(this.maxDelayMs, 1000 * 2 ** (this._failures - 1))
      + Math.random() * 500; // jitter — don't stampede a recovering server
    this._timer = setTimeout(() => this._connect(), delay);
  }
}
