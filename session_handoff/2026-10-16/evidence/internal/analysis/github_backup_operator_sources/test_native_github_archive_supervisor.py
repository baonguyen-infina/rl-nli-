"""Regression checks for preserving logs between model checkpoints."""
import unittest
import native_github_archive_supervisor as supervisor

class LogBackupTests(unittest.TestCase):
    def event(self,observation,now,last_log_at,seen=None):
        policy=getattr(supervisor,'log_event',None)
        self.assertIsNotNone(policy,'A logs-only backup policy is required')
        return policy(observation,now,last_log_at,seen)

    def test_periodic_log_backup_does_not_require_a_checkpoint(self):
        self.assertEqual(self.event({'ready':[]},1000,0),'logs-1000')

    def test_failure_logs_are_backed_up_even_after_a_recent_snapshot(self):
        for stage in ['training_failed','test_failed']:
            self.assertEqual(self.event({'failure_stage':stage},1000,990),'logs-1000')

    def test_unchanged_failure_does_not_trigger_duplicate_snapshot_each_poll(self):
        self.assertIsNone(self.event({'failure_stage':'test_failed'},1000,990,'test_failed'))

    def test_healthy_job_waits_for_next_log_interval(self):
        self.assertIsNone(self.event({'ready':['checkpoint-100']},1000,990))

if __name__=='__main__':unittest.main()
