import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manage", ROOT / "scripts" / "manage.py")
manage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage)


class ManageCliChecks(unittest.TestCase):
    def test_missing_tool_error_is_actionable(self):
        with patch.object(manage.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                manage.ensure_command("oc")
        self.assertIn("oc", str(ctx.exception).lower())
        self.assertIn("crc", str(ctx.exception).lower())

    def test_render_uses_reduced_poc_memory_profile(self):
        output_dir = manage.generate(manage.config())
        resources = manage.yaml.safe_load_all((output_dir / "app.yaml").read_text())
        deployment = next(item for item in resources if item["kind"] == "Deployment")
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        args = container["args"]

        self.assertEqual(container["resources"]["requests"]["memory"], "6Gi")
        self.assertEqual(container["resources"]["limits"]["memory"], "6Gi")
        self.assertEqual(args[args.index("--max-model-len") + 1], "128")
        self.assertEqual(args[args.index("--kv-cache-memory-bytes") + 1], "6291456")
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "1")
        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "128")
        env = {item["name"]: item["value"] for item in container["env"]}
        self.assertNotIn("VLLM_CPU_KVCACHE_SPACE", env)
        self.assertEqual(env["VLLM_NO_USAGE_STATS"], "1")


if __name__ == "__main__":
    unittest.main()
