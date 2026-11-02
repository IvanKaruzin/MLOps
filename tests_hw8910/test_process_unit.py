"""Additional process integrity checks; no models or orchestration server."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lab
import release


class ProcessIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'workspace'
        lab.initialize(self.root)

    def test_artifact_bytes_and_aggregate_are_checked_before_publication(self):
        manifest = lab.read(self.root / 'manifest-B.json')
        release.verify_artifact(manifest, self.root)
        original = (self.root / 'accepted.json').read_bytes()
        (self.root / 'model-B.txt').write_text('changed artifact', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            lab.run(self.root, 'report-B.json')
        self.assertEqual((self.root / 'accepted.json').read_bytes(), original)
        self.assertFalse((self.root / 'publications.jsonl').exists())

    def test_artifact_paths_cannot_escape_by_parent_absolute_or_symlink(self):
        manifest = lab.read(self.root / 'manifest-B.json')
        outside = self.root.parent / 'outside.txt'
        outside.write_text('fixture', encoding='utf-8')
        (self.root / 'escape.txt').symlink_to(outside)
        for name in ('../outside.txt', str(outside), 'escape.txt', 'sub\\file'):
            changed = copy.deepcopy(manifest)
            changed['file_hashes'] = {name: release.file_sha256(outside)}
            changed['model_sha256'] = release.digest(release.canonical(changed['file_hashes']))
            with self.subTest(name=name), self.assertRaises(ValueError):
                release.verify_artifact(changed, self.root)

    def test_fsync_error_preserves_previous_pointer_and_cleans_temporary(self):
        manifest = lab.read(self.root / 'manifest-B.json')
        report = lab.read(self.root / 'report-B.json')
        target = self.root / 'accepted.json'
        before = target.read_bytes()
        with patch('release.os.fsync', side_effect=OSError('simulated fsync failure')):
            with self.assertRaises(OSError):
                release.publish(manifest, report, target)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_persisted_preparation_is_verified_and_uses_content_not_path(self):
        prepared = lab.prepare(self.root)
        self.assertEqual(lab.prepare(self.root), prepared)
        cache_file = self.root / 'cache' / f"{prepared['cache_key']}.json"
        cache = lab.read(cache_file)
        cache['result']['rows'] = 999
        lab.write(cache_file, cache)
        with self.assertRaisesRegex(ValueError, 'Corrupted persisted'):
            lab.prepare(self.root)

    def test_pending_journal_record_cannot_be_rebound_after_pointer_crash(self):
        original_publish = release.publish
        def fail_after_pointer(*args, **kwargs):
            original_publish(*args, **kwargs)
            raise RuntimeError('injected after pointer')
        with patch.object(release, 'publish', side_effect=fail_after_pointer):
            with self.assertRaises(RuntimeError):
                lab.run(self.root, 'report-B.json')
        pending_path = self.root / 'publication-intent.json'
        pending = lab.read(pending_path)
        pending['record']['model_id'] = 'unrelated-model'
        lab.write(pending_path, pending)
        before = (self.root / 'accepted.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Pending journal record'):
            lab.run(self.root, 'report-B.json')
        self.assertEqual((self.root / 'accepted.json').read_bytes(), before)
        self.assertFalse((self.root / 'publications.jsonl').exists())

    def test_arbitrarily_large_integer_metrics_raise_value_error(self):
        manifest = lab.read(self.root / 'manifest-B.json')
        for key in ('accuracy', 'format_rate'):
            report = lab.read(self.root / 'report-B.json')
            report['metrics'][key] = 10 ** 1000
            with self.subTest(metric=key), self.assertRaises(ValueError):
                release.validate_release(manifest, report)


if __name__ == '__main__':
    unittest.main()
