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

if __name__ == '__main__':
    unittest.main()
