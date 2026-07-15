// Shared cross-module state. Only state that genuinely crosses feature
// boundaries lives here — feature-local state (timers, buffers, deck
// position…) stays private inside its owning module.
export const store = {
  currentTab: 'url',
  selectedMode: 'gemini_direct',
  currentTaskId: null,        // id of the task the live view is attached to
  currentResult: null,        // completed LessonResult shown in the results card
  currentSource: '',          // URL or filename — used by the exporters
  processingStartTime: null,  // ms epoch — drives the ETA estimate
  taskFinished: false,        // guards late events after a terminal state
};
