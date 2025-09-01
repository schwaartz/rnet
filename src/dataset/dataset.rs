use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

const RANDSEED: u64 = 0;

/// A useful wrapper struct around a pair of input and output data
pub struct Dataset {
    pub inputs: Vec<Array1<f32>>,
    pub outputs: Vec<Array1<f32>>,
}

impl Dataset {
    /// Creates a new Dataset by consuming the input and output data vectors
    pub fn new(inputs: Vec<Array1<f32>>, outputs: Vec<Array1<f32>>) -> Self {
        assert!(
            inputs.len() == outputs.len(),
            "Input and output vectors must have the same length ({} != {})",
            inputs.len(),
            outputs.len(),
        );
        Self { inputs, outputs }
    }

    /// Returns the number of samples in the dataset
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Returns a regular iterator over all the inputs and outputs
    pub fn iterator(&self) -> impl Iterator<Item = (&Array1<f32>, &Array1<f32>)> {
        self.inputs.iter().zip(self.outputs.iter())
    }

    /// Returns an iterator over random batches from the dataset with the given size.
    /// The last elements are returned in a batch that is possibly smaller.
    pub fn random_iterator(&self, batch_size: usize) -> impl Iterator<Item = Vec<(&Array1<f32>, &Array1<f32>)>> {
        let mut rng = StdRng::seed_from_u64(RANDSEED);
        let mut indices: Vec<usize> = (0..self.inputs.len()).collect();
        indices.shuffle(&mut rng);

        let batches: Vec<Vec<usize>> = indices.chunks(batch_size).map(|chunk| chunk.to_vec()).collect();
        batches.into_iter().map(move |batch| {
            batch.into_iter().map(|i| (&self.inputs[i], &self.outputs[i])).collect()
        })
    }

    /// Splits the dataset into two new datasets with the given ratio. E.g. when
    /// the ratio is 0.8, the first dataset will have +- 80% of the values and
    /// the second dataset will have the remaining 20%. The object on which the
    /// method is called is consumed in the process. Additionally, the split is
    /// random, so that this method can be used to split a dataset into a training
    /// and testing set.
    pub fn split(self, ratio: f32) -> (Dataset, Dataset) {
        assert!(ratio > 0.0 && ratio < 1.0, "Ratio ({}) must be a part of the open interval (0.0, 1.0)", ratio);

        let split_index = (self.inputs.len() as f32 * ratio).round() as usize;
        let mut rng = StdRng::seed_from_u64(RANDSEED);
        let mut indices: Vec<usize> = (0..self.inputs.len()).collect();
        indices.shuffle(&mut rng);
        let (train_indices, test_indices) = indices.split_at(split_index);
        (
            Dataset::new(
                train_indices.iter().map(|&i| self.inputs[i].clone()).collect(),
                train_indices.iter().map(|&i| self.outputs[i].clone()).collect(),
            ),
            Dataset::new(
                test_indices.iter().map(|&i| self.inputs[i].clone()).collect(),
                test_indices.iter().map(|&i| self.outputs[i].clone()).collect(),
            ),
        )
    }
}
#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::*;

    fn create_sample_dataset() -> Dataset {
        let inputs = vec![
            array![1.0, 2.0],
            array![3.0, 4.0],
            array![5.0, 6.0],
            array![7.0, 8.0],
        ];
        let outputs = vec![
            array![10.0],
            array![20.0],
            array![30.0],
            array![40.0],
        ];
        Dataset::new(inputs, outputs)
    }

    #[test]
    fn test_iterator() {
        let dataset = create_sample_dataset();
        let pairs: Vec<_> = dataset.iterator().collect();
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0].0, &array![1.0, 2.0]);
        assert_eq!(pairs[0].1, &array![10.0]);
    }

    #[test]
    fn test_random_iterator_batches() {
        let dataset = create_sample_dataset();
        let batches: Vec<_> = dataset.random_iterator(2).collect();

        // There should be 2 batches (4 elements, batch size 2)
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|batch| batch.len() == 2));
    }

    #[test]
    fn test_random_iterator_last_batch_smaller() {
        let dataset = create_sample_dataset();
        let batches: Vec<_> = dataset.random_iterator(3).collect();

        // 4 elements, batch size 3 => 2 batches (3 + 1)
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn test_split_ratio() {
        let dataset = create_sample_dataset();
        let (train, test) = dataset.split(0.75);

        // 4 elements, 0.75 ratio => 3 train, 1 test
        assert_eq!(train.inputs.len(), 3);
        assert_eq!(test.inputs.len(), 1);
    }

    #[test]
    fn test_split_reproducibility() {
        let dataset1 = create_sample_dataset();
        let dataset2 = create_sample_dataset();
        let (train1, test1) = dataset1.split(0.5);
        let (train2, test2) = dataset2.split(0.5);

        // With the same seed, splits should be identical
        assert_eq!(train1.inputs, train2.inputs);
        assert_eq!(test1.inputs, test2.inputs);
    }

    #[test]
    fn test_len() {
        let dataset = create_sample_dataset();
        assert_eq!(dataset.len(), 4);
    }

    #[test]
    #[should_panic(expected = "Ratio (1) must be a part of the open interval (0.0, 1.0)")]
    fn test_split_invalid_ratio() {
        let dataset = create_sample_dataset();
        dataset.split(1.0);
    }

    #[test]
    #[should_panic(expected = "Input and output vectors must have the same length (2 != 3)")]
    fn test_dataset_mismatch() {
        let inputs = vec![array![1.0, 2.0], array![3.0, 4.0]];
        let outputs = vec![array![10.0], array![20.0], array![30.0]];
        Dataset::new(inputs, outputs);
    }
}
