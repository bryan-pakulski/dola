// Expects a folder with subfolders containing different image classes, subfolder names are used as label names
pub trait ClassificationFolderLoader<T> {
    fn load(&self, path: &str) -> Self;
    fn shuffle(&self);
    fn get(&self, index: usize) -> Option<(&T, &String)>;
    fn next(&mut self) -> Option<(&T, &String)>;
}

pub struct Loader<T> {
    data: Vec<(T, String)>,
    map: Vec<usize>,
    iter: usize,
}

impl<T> ClassificationFolderLoader<T> for Loader<T> {
    fn load(&self, path: &str) -> Self {
        let data_loader: Loader<T> = Loader {
            data: vec![],
            map: vec![0; self.data.len()],
            iter: 0,
        };

        // Load data from subfolders, create a image and label pair

        data_loader
    }
    fn shuffle(&self) {
        // TODO: shuffle the map which we use to index the data
    }

    fn get(&self, index: usize) -> Option<(&T, &String)> {
        match self.map.get(index) {
            Some(idx) => self.data.get(*idx).map(|(image, label)| (image, label)),
            None => None,
        }
    }

    fn next(&mut self) -> Option<(&T, &String)> {
        if self.iter < self.map.len() {
            let idx = self.map[self.iter];
            let data = self.get(idx);
            self.iter += 1;
            data
        } else {
            None
        }
    }
}
