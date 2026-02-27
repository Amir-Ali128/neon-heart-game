import io
import unittest

from app import app


class NeuroVisionPipelineApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_pipeline_requires_image(self):
        response = self.client.post("/api/neurovision/run", data={})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("error", payload)

    def test_pipeline_returns_structured_json(self):
        data = {"image": (io.BytesIO(b"fake-image-content"), "test.png")}
        response = self.client.post(
            "/api/neurovision/run",
            data=data,
            headers={"X-Request-Id": "req-123"},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)

        payload = response.get_json()
        self.assertEqual(payload["metadata"]["request_id"], "req-123")
        self.assertIn("detections", payload)
        self.assertIn("semantic", payload)
        self.assertIn("explainability_heatmap", payload)
        self.assertIn("activations", payload)
        self.assertIn("network_state_3d", payload)


if __name__ == "__main__":
    unittest.main()
