/*
 * This code defines a struct HMM to represent a hidden Markov model.
 * It implements the viterbi function to compute the most likely sequence of states given a sequence of observations. The function takes a reference to a vector of observations and returns a vector of state indices representing the most likely path. The implementation uses dynamic programming to compute the maximum probability and the corresponding backpointer at each time step and backtracks from the final state to find the most likely path. 
 * It also implements forward and backward function to computate the probablities of ending in a specific state seeing a part of a sequence.
 * At last, it implements the Baum-Welch training, which uses the previous functions.
 */


use std::f64;

//function for summing log probabilities, given vector
fn log_sum_exp_slice(slice: &[f64]) -> f64 {
    let max_val = *slice.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    let sum: f64 = slice.iter().map(|x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

// Define a struct to represent a hidden Markov model
struct HMM {
    num_states: usize,
    num_obs: usize,
    initial_log_probs: Vec<f64>,
    transition_log_probs: Vec<Vec<f64>>,
    emission_log_probs: Vec<Vec<f64>>,
    end_log_probs: Vec<f64>,
}

impl HMM {
    //define new markov model 
    fn new( initial_log_probs: Vec<f64>, transition_log_probs: Vec<Vec<f64>>, emission_log_probs: Vec<Vec<f64>>) -> Self {
        let num_states = initial_log_probs.len();
        let num_obs = emission_log_probs[0].len();
        let end_log_probs =  vec![0.0; initial_log_probs.len()];
        HMM {num_states, num_obs, initial_log_probs, transition_log_probs, emission_log_probs, end_log_probs}
    }
    
    //define new markov model 
    fn new_with_end( initial_log_probs: Vec<f64>, transition_log_probs: Vec<Vec<f64>>, emission_log_probs: Vec<Vec<f64>>, end_log_probs: Vec<f64>) -> Self {
        let num_states = initial_log_probs.len();
        let num_obs = emission_log_probs[0].len();
        HMM {num_states, num_obs, initial_log_probs, transition_log_probs, emission_log_probs, end_log_probs}
    }
    
    // Function to compute the log probability of an observation sequence using the forward algorithm
    fn forward(&self, observations: &[usize]) -> Vec<Vec<f64>>  {
        let mut alpha = vec![vec![0.0; self.num_states]; observations.len()];
        // Initialize the first column of the alpha matrix
        for s in 0..self.num_states {
            alpha[0][s] = self.initial_log_probs[s] + self.emission_log_probs[s][observations[0]];
        }
        
        // Iterate over the remaining columns of the alpha matrix
        for t in 1..observations.len() {
            
            for s in 0..self.num_states {
                alpha[t][s] = log_sum_exp_slice(&(0..self.num_states)
                    .map(|prev_s| {
                        alpha[t - 1][prev_s] + self.transition_log_probs[prev_s][s] +
                            self.emission_log_probs[s][observations[t]]
                    })
                    .collect::<Vec<f64>>());
            }
        }
        
        alpha        
    }

   // Function to compute the log probability of an observation sequence using the backward algorithm
    //returns the matrice of all probabilities
   fn backward(&self, observations: &[usize]) -> Vec<Vec<f64>> {
        let mut beta = vec![vec![0.0; self.num_states]; observations.len()];
        
        // Initialize the last column of the beta matrix with end probabilites
        for s in 0..self.num_states {
            beta[observations.len() - 1][s] = self.end_log_probs[s];             
        }
        
        // Iterate over the remaining columns of the beta matrix
        for t in (1..observations.len()).rev() {            
            for s in 0..self.num_states {
                beta[t-1][s] = log_sum_exp_slice(&(0..self.num_states)
                    .map(|next_s| {
                        beta[t][next_s] + self.transition_log_probs[s][next_s] +
                            self.emission_log_probs[next_s][observations[t]]
                    })
                    .collect::<Vec<f64>>());
            }
        }
        
        beta
    }

    // Function to compute likelihood of every path given the observation sequence
    // returns the trellis matrix and backpointers matrix
    fn viterbi(&self, observations: &[usize]) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
        let mut trellis = vec![vec![f64::NEG_INFINITY; self.num_states]; observations.len()];
        let mut backpointers = vec![vec![0; self.num_states]; observations.len()];
        
        // Initialize the first column of the trellis matrix
        for s in 0..self.num_states {
            trellis[0][s] = self.initial_log_probs[s] + self.emission_log_probs[s][observations[0]];
            backpointers[0][s] = s;
        }
        
        // Iterate over the remaining columns of the trellis matrix
        for t in 1..observations.len() {
            for s in 0..self.num_states {
                let (max_log_prob, max_prev_s) = (0..self.num_states)
                    .map(|prev_s| {
                        let prob = trellis[t - 1][prev_s] + self.transition_log_probs[prev_s][s] +
                            self.emission_log_probs[s][observations[t]];
                        (prob, prev_s)
                    })
                    .max_by(|&(p1, _), &(p2, _)| p1.partial_cmp(&p2).unwrap())
                    .unwrap();
                    
                trellis[t][s] = max_log_prob;
                backpointers[t][s] = max_prev_s;
                
                if t == observations.len()-1 {
                    trellis[t][s] += self.end_log_probs[s];
                }
            }
        }
        
        (trellis, backpointers)
    }
    
    // Function to compute the most likely sequence of states and probabilities of all possible paths
    // and also returns the log probabilities of the most probable path
    fn viterbi_max(&self, observations: &[usize]) -> (Vec<usize>, f64, Vec<f64>) {
        let (trellis, backpointers) = self.viterbi(&observations);
        
        // Find the state with the highest log probability in the last column of the trellis matrix
        let (mut max_log_prob, mut max_state) = (f64::NEG_INFINITY, 0);
        for s in 0..self.num_states {
            if trellis[observations.len() - 1][s] > max_log_prob {
                max_log_prob = trellis[observations.len() - 1][s];
                max_state = s;
            }
        }
        
        // Backtrack through the backpointers to find the most likely state sequence
        let mut state_sequence = vec![max_state; observations.len()];
        for t in (1..observations.len()).rev() {
            state_sequence[t - 1] = backpointers[t][state_sequence[t]];
        }
        
        // Compute the log probability of all paths using the forward algorithm
        let log_probs = (0..self.num_states)
            .map(|s| {
                let mut log_prob = self.initial_log_probs[s] + self.emission_log_probs[s][observations[0]];
                for t in 1..observations.len() {
                    log_prob += self.transition_log_probs[state_sequence[t - 1]][state_sequence[t]] +
                        self.emission_log_probs[state_sequence[t]][observations[t]];
                }
                log_prob
            })
            .collect();
            
        (state_sequence, max_log_prob, log_probs)
    }
    
    
    // Function to run the Baum-Welch algorithm and update the model parameters
    fn baum_welch(&mut self, observations: &[usize], max_iterations: usize) {
        for _ in 0..max_iterations {
            
            // Compute alpha and beta matrices using current model parameters
            let alpha = self.forward(&observations);
            let beta = self.backward(&observations);
            
            // Compute gamma and xi matrices using alpha and beta matrices
            let mut gamma = vec![vec![0.0; self.num_states]; observations.len()];
            let mut xi = vec![vec![vec![0.0; self.num_states]; self.num_states]; observations.len() - 1];
            
            for t in 0..observations.len() {                     
                let normalizer = log_sum_exp_slice(&alpha[observations.len()-1]);
                for s in 0..self.num_states {
                    gamma[t][s] = alpha[t][s] + beta[t][s] - normalizer;
                }
                
                if t < observations.len() - 1 {                
                    for s in 0..self.num_states {
                        for next_s in 0..self.num_states {
                            xi[t][s][next_s] = alpha[t][s] 
                                + self.transition_log_probs[s][next_s]
                                + self.emission_log_probs[next_s][observations[t+1]]
                                + beta[t + 1][next_s] 
                                - normalizer;
                        }
                    }
                }
            }
            
            // Update initial probabilities            
            for s in 0..self.num_states {
                self.initial_log_probs[s] = gamma[0][s];
            }
            
            // Update transition probabilities
            for s in 0..self.num_states {
                
                let normalizer = log_sum_exp_slice(&(0..observations.len()-1)
                                .map(|a| log_sum_exp_slice(&xi[a][s]))
                                .collect::<Vec<f64>>());
                
                for next_s in 0..self.num_states {
                    let numerator = log_sum_exp_slice(&xi.iter().map(|xi_t| xi_t[s][next_s]).collect::<Vec<f64>>());
                    self.transition_log_probs[s][next_s] = numerator - normalizer;
                }
            }
            
            // Update emission probabilities
            for s in 0..self.num_states {
                let normalizer = log_sum_exp_slice(&gamma.iter().map(|g| g[s]).collect::<Vec<f64>>());
                
                for o in 0..self.num_obs {
                    let numerator = log_sum_exp_slice(&observations.iter().enumerate().filter(|(_, &obs)| obs == o).map(|(t, _)| gamma[t][s]).collect::<Vec<f64>>());
                    self.emission_log_probs[s][o] = numerator - normalizer;
                }
            }
        }
    }
}

//simple tests and example usage for viterbi, forward, backward and baum_welch function

#[cfg(test)]
 mod tests {
    use super::*;
    
    /* pi is inital probabilities matrix
     * a is transtition probabilities matrix
     * b is emission probabilities matrix 
     * omega is end probabilities matrix*/
    
    
    #[test]
    fn test_viterbi1() {
        
        // We construct an example from Borodovsky & Ekisheva (2006), pp. 80.
        // http://cecas.clemson.edu/~ahoover/ece854/refs/Gonze-ViterbiAlgorithm.pdf
        // States: 0 = (H) High GC content, 1 = (L) Low GC content
        // Symbols: 0 = A, 1 = C, 2 = G, 3 = T
        let pi = vec![0.5f64.ln(), 0.5f64.ln()];
        
        let a = vec![
            vec![0.5f64.ln(), 0.5f64.ln()], 
            vec![0.4f64.ln(), 0.6f64.ln()]
        ];
        
        let b = vec![
            vec![0.2f64.ln(), 0.3f64.ln(), 0.3f64.ln(), 0.2f64.ln()], 
            vec![0.3f64.ln(), 0.2f64.ln(), 0.2f64.ln(), 0.3f64.ln()]
        ];

        let hmm = HMM::new(pi, a, b);
        let (path, prob, _) = hmm.viterbi_max(&[2, 2, 1, 0, 1, 3, 2, 0, 0]);

        let expected = vec![0, 0, 0, 1, 1, 1, 1, 1, 1];
        
        assert_eq!(expected, path);
        
        assert!((4.25e-8_f64.ln() - prob).abs() < 0.001);
    }
    
    #[test]
    //testing zero probabilities and end probabilities
    fn test_viterbi2() {
        let pi = vec![0.34f64.ln(), 0.33f64.ln(), 0.33f64.ln()];
        
        let a = vec![
            vec![0.7f64.ln(), 0.3f64.ln(), 0.0f64.ln()], 
            vec![0.0f64.ln(), 0.4f64.ln(), 0.6f64.ln()],
            vec![0.5f64.ln(), 0.0f64.ln(), 0.5f64.ln()]
        ];
        
        let b = vec![
            vec![0.3f64.ln(), 0.7f64.ln()], 
            vec![0.9f64.ln(), 0.1f64.ln()], 
            vec![0.4f64.ln(), 0.6f64.ln()]
        ];
        
        let omega = vec![0.2f64.ln(), 0.8f64.ln(), 0.0f64.ln()];

        let hmm = HMM::new_with_end(pi, a, b, omega);
        let (path, prob, _) = hmm.viterbi_max(&[1, 0, 0]);
        
        let expected = vec![0, 1, 1];
        
        assert_eq!(expected, path);
        
        assert!((185e-4_f64.ln() - prob).abs() < 0.001);
    }
    
    #[test]
    fn test_forward_equals_backward() {
        
        // Same example as above from the same article
        let pi = vec![0.5f64.ln(), 0.5f64.ln()];
        let a = vec![
            vec![0.5f64.ln(), 0.5f64.ln()], 
            vec![0.4f64.ln(), 0.6f64.ln()]
        ];
        let b = vec![
            vec![0.2f64.ln(), 0.3f64.ln(), 0.3f64.ln(), 0.2f64.ln()], 
            vec![0.3f64.ln(), 0.2f64.ln(), 0.2f64.ln(), 0.3f64.ln()]
        ];

        
        let hmm = HMM::new(pi, a, b);
        
        for i in 1..10 {
            let mut seq: Vec<usize> = vec![0; i];
            while seq.iter().sum::<usize>() != i {
                for j in 0..i {
                    if seq[j] == 0 {
                        seq[j] = 1;
                        break;
                    } else {
                        seq[j] = 0;
                    }
                }
                let forward_prob = log_sum_exp_slice(&hmm.forward(&seq)[i-1]);
                let backward = &hmm.backward(&seq)[0];
                let backward_prob = log_sum_exp_slice(&(0..hmm.num_states)
                    .map(|s| backward[s] + hmm.initial_log_probs[s] + hmm.emission_log_probs[s][seq[0]])
                    .collect::<Vec<f64>>()
                );
                assert!((forward_prob - backward_prob).abs() < 0.0001);
            }
        }
    }
    
    #[test]
    fn test_baum_welch() {
        let pi = vec![0.2f64.ln(), 0.8f64.ln()];
        let a = vec![
            vec![0.5f64.ln(), 0.5f64.ln()], 
            vec![0.3f64.ln(), 0.7f64.ln()]
        ];
        let b = vec![
            vec![0.3f64.ln(), 0.7f64.ln()], 
            vec![0.8f64.ln(), 0.2f64.ln()]
        ];

        let mut hmm = HMM::new(pi.clone(), a.clone(), b.clone());
        
        let obs = vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 0];
        
        hmm.baum_welch(&obs, 1);
        
        let test_pi = vec![0.0719f64.ln(), 0.9281f64.ln()];
        let test_a = vec![0.4392f64.ln(), 0.5608f64.ln(), 0.2145f64.ln(), 0.7855f64.ln()];
        let test_b = vec![0.4616f64.ln(), 0.5384f64.ln(), 0.9150f64.ln(), 0.0850f64.ln(), 0.4251f64.ln(), 0.5165f64.ln()];
        
        let flatten_pi = hmm.initial_log_probs.clone();
        let flatten_a = hmm.transition_log_probs
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let flatten_b = hmm.emission_log_probs
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
           
        assert!(flatten_pi
            .iter()
            .zip(test_pi.iter())
            .all(|(x, y)| (x-y).abs() < 0.001)
        );
        
        assert!(flatten_a
            .iter()
            .zip(test_a.iter())
            .all(|(x, y)| (x-y).abs() < 0.001)
        );
        
        assert!(flatten_b
            .iter()
            .zip(test_b.iter())
            .all(|(x, y)| (x-y).abs() < 0.001)
        );
    }
    
    #[test]
    fn test_baum_welch2() {
        let pi = vec![0.3f64.ln(), 0.7f64.ln()];
        let a = vec![
            vec![0.8f64.ln(), 0.1f64.ln()], 
            vec![0.1f64.ln(), 0.8f64.ln()]
        ];
        let b = vec![
            vec![0.7f64.ln(), 0.2f64.ln(), 0.1f64.ln()], 
            vec![0.1f64.ln(), 0.2f64.ln(), 0.7f64.ln()]
        ];

        let mut hmm = HMM::new(pi, a, b);
        
        let obs = vec![
            1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 2, 2,
            1, 2, 1, 1,
        ];
        
        hmm.baum_welch(&obs, 1);
        
        let test_pi = vec![0.0597f64.ln(), 0.9403f64.ln()];
        let test_a = vec![0.8935f64.ln(), 0.1065f64.ln(), 0.0962f64.ln(), 0.9038f64.ln()];
        let test_b = vec![0.6805f64.ln(), 0.2152f64.ln(), 0.1043f64.ln(), 0.0581f64.ln(), 0.4269f64.ln(), 0.5149f64.ln()];
        
        let flatten_pi = hmm.initial_log_probs.clone();
        let flatten_a = hmm.transition_log_probs
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
        let flatten_b = hmm.emission_log_probs
            .into_iter()
            .flatten()
            .collect::<Vec<f64>>();
        
            
         assert!(flatten_pi
            .iter()
            .zip(test_pi.iter())
            .all(|(x, y)| (x-y).abs() < 0.001)
        );
        
        assert!(flatten_a
            .iter()
            .zip(test_a.iter())
            .all(|(x, y)| (x-y).abs() < 0.001)
        );
        
        assert!(flatten_b
            .iter()
            .zip(test_b.iter())
            .all(|(x, y)| (x-y).abs() < 0.001)
        );
    }
}

fn main() {    
    test_viterbi1();
    test_viterbi2();
    test_forward_equals_backward();
    test_baum_welch();
    test_baum_welch2();
    println!("Run successfully!");
}



