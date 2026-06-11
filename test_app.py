import unittest
import json
from app import app

class TestSurgicalVisionAI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_catalog_endpoint(self):
        """
        Verify the GET /api/catalog endpoint returns all items and correct properties.
        """
        print("\nTesting /api/catalog endpoint...")
        response = self.app.get('/api/catalog')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        print(f"Success: Catalog returned {len(data)} items.")
        
        # Verify schema
        first_item = data[0]
        self.assertIn('id', first_item)
        self.assertIn('name', first_item)
        self.assertIn('sku', first_item)
        self.assertIn('category', first_item)
        self.assertIn('page', first_item)
        self.assertIn('image_path', first_item)

    def test_classify_endpoint(self):
        """
        Verify the POST /api/classify endpoint with a real instrument image file.
        """
        print("\nTesting /api/classify endpoint with a sample instrument image...")
        
        # We will use one of our cropped catalog images as the mock upload file
        sample_img_path = 'dataset/processed/p02_r1_c5.png'
        
        with open(sample_img_path, 'rb') as img_file:
            response = self.app.post(
                '/api/classify',
                data={
                    'image': (img_file, 'test_image.png')
                },
                content_type='multipart/form-data'
            )
            
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertTrue(data.get('success'))
        results = data.get('results')
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        print("Matches found:")
        for idx, match in enumerate(results[:3]):
            print(f"Rank {idx+1}: {match['name']} | Similarity: {match['similarity']*100:.2f}%")
            
        # The top match should be p02_r1_c5 itself!
        top_match = results[0]
        self.assertEqual(top_match['id'], 'p02_r1_c5')
        print(f"Success: Top match matched query item correctly with {top_match['similarity']*100:.2f}% similarity.")

    def test_settings_endpoints(self):
        """Verify GET and POST /api/settings."""
        print("\nTesting /api/settings endpoints...")
        
        # Test GET settings
        response = self.app.get('/api/settings')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('has_api_key', data)
        
        # Test POST settings with groq key
        response = self.app.post(
            '/api/settings',
            data=json.dumps({'groq_api_key': 'valid_groq_key_for_tests'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get('success'))
        
        # Check settings again — has_api_key should reflect the active key
        response = self.app.get('/api/settings')
        data = json.loads(response.data)
        self.assertTrue(data.get('has_groq_key'))
        self.assertTrue(data.get('has_api_key'))

    def test_upload_image_endpoint(self):
        """Verify POST /api/upload-image saves the image."""
        print("\nTesting /api/upload-image endpoint...")
        sample_img_path = 'dataset/processed/p02_r1_c5.png'
        
        with open(sample_img_path, 'rb') as img_file:
            response = self.app.post(
                '/api/upload-image',
                data={
                    'image': (img_file, 'test_manual.png')
                },
                content_type='multipart/form-data'
            )
            
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get('success'))
        self.assertIn('raw_image_path', data)
        
        # Verify file path is in raw folder
        raw_path = data.get('raw_image_path')
        self.assertTrue(raw_path.startswith('dataset/raw/manual_'))

    def test_upload_catalog_endpoint(self):
        """Verify POST /api/upload-catalog returns a job_id."""
        print("\nTesting /api/upload-catalog endpoint...")
        import io
        
        # Set a test API key to pass check
        response = self.app.post(
            '/api/settings',
            data=json.dumps({'groq_api_key': 'valid_groq_key_for_tests'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        pdf_data = b"%PDF-1.4\n%mock pdf content\n%%EOF"
        response = self.app.post(
            '/api/upload-catalog',
            data={
                'catalog': (io.BytesIO(pdf_data), 'test_catalog.pdf')
            },
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('job_id', data)
        self.assertIn('filename', data)
        
        job_id = data.get('job_id')
        
        # Test pipeline status polling
        response = self.app.get(f'/api/pipeline-status/{job_id}')
        self.assertEqual(response.status_code, 200)
        status_data = json.loads(response.data)
        self.assertIn('stage', status_data)
        self.assertIn('progress_pct', status_data)

if __name__ == '__main__':
    unittest.main()
