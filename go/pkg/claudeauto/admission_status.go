package claudeauto

// RouterAdmissionStatus is an in-memory observation of main-model admission.
// Reading it never polls a backend or adds a scheduler task.
type RouterAdmissionStatus struct {
	Active, Queued, Limit int64
}

func (r *Router) AdmissionStatus() RouterAdmissionStatus {
	if r == nil {
		return RouterAdmissionStatus{}
	}
	return RouterAdmissionStatus{
		Active: r.mainActive.Load(),
		Queued: r.mainQueued.Load(),
		Limit:  int64(r.maxMainActive),
	}
}
