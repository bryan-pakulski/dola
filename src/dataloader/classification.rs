use glob::glob;
use std::collections::HashMap;

use rand::rng;
use rand::seq::SliceRandom;

// Expects a folder with subfolders containing different image classes, subfolder names are used as label names
pub trait ClassificationFolderLoader {
    fn load(&mut self, path: &str);
    fn shuffle(&mut self);
    fn get(&self, index: usize) -> Option<(Vec<f32>, Vec<f32>)>;
    fn next(&mut self) -> Option<(Vec<f32>, Vec<f32>)>;
    fn load_image(&self, path: &str) -> Vec<f32>;
}

pub struct Loader {
    images: Vec<(String, usize)>,
    index_order: Vec<usize>,
    label_map: HashMap<String, usize>,
    label_count: usize,
    iter: usize,
}

impl Loader {
    pub fn new() -> Loader {
        let data_loader: Loader = Loader {
            images: vec![],
            index_order: vec![0; 0],
            label_map: vec![].into_iter().collect(),
            iter: 0,
            label_count: 0,
        };

        data_loader
    }

    pub fn size(&self) -> usize {
        self.images.len()
    }
}

impl ClassificationFolderLoader for Loader {
    fn load(&mut self, path: &str) {
        // Load data from subfolders, create a image and label pair
        let blob_query = path.to_owned() + "/**/*.jpg";
        for entry in glob(blob_query.as_str()).expect(
            format!(
                "Failed to read glob pattern: {}, unable to load dataset: {}",
                blob_query, path
            )
            .as_str(),
        ) {
            match entry {
                Ok(path) => match path.parent() {
                    Some(parent) => {
                        let label = parent.file_name().unwrap().to_str().unwrap();
                        let label_idx: usize;

                        if self.label_map.contains_key(label) {
                            label_idx = *self.label_map.get(label).unwrap();
                        } else {
                            label_idx = self.label_count;
                            self.label_count += 1;
                        }

                        self.label_map.insert(label.to_string(), label_idx);
                        self.images
                            .push((path.to_str().unwrap().to_string(), label_idx));
                    }
                    None => {
                        println!(
                            "Error: No parent folder found for {}",
                            path.to_str().unwrap()
                        );
                    }
                },
                Err(e) => println!("Error: {}", e),
            }
        }

        println!("Loaded {} images", self.images.len());
        println!("Loaded {} labels", self.label_count);
        self.index_order = (0..self.images.len()).collect();
    }

    fn shuffle(&mut self) {
        let mut rng = rng();
        self.index_order.shuffle(&mut rng);
    }

    fn load_image(&self, path: &str) -> Vec<f32> {
        let mut img_data: Vec<f32> = vec![];

        let img = image::open(path).unwrap();
        let raw_img = img.to_luma32f().into_raw();

        for i in 0..raw_img.len() {
            img_data.push((raw_img[i] / 255.0).into());
        }

        img_data
    }

    fn get(&self, index: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        match self.index_order.get(index) {
            Some(idx) => {
                let metadata: &(String, usize) = self.images.get(*idx).unwrap();
                let img_path = metadata.0.as_str();
                let label = metadata.1;
                let img_data: Vec<f32> = self.load_image(img_path);

                let mut output_data: Vec<f32> = vec![0.0f32; self.label_count];
                output_data[label] = 1.0f32;

                Some((img_data, output_data))
            }
            None => None,
        }
    }

    fn next(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        if self.iter < self.index_order.len() {
            let idx = self.index_order[self.iter];
            let data = self.get(idx);
            self.iter += 1;
            data
        } else {
            self.iter = 0;
            None
        }
    }
}
