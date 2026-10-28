package update

// A Windows update renames the running ggrun.exe to ggrun.exe.old because a
// running executable cannot be overwritten. Remove it on the next start.
func init() {
	RemoveReplacedBinary()
}
