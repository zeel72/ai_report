"""
Gaussian Hidden Markov Model Implementation
Lab Assignment 5: Gaussian Hidden Markov Models for Financial Time Series Analysis
Course: CS307 - Artificial Intelligence
Week 5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import warnings
from scipy import stats
import itertools
from sklearn.metrics import silhouette_score
import pickle

warnings.filterwarnings('ignore')

class GaussianHMM:
    """
    Gaussian Hidden Markov Model for Financial Time Series Analysis
    """
    
    def __init__(self, n_states=2, random_state=42):
        """
        Initialize Gaussian HMM
        
        Args:
            n_states (int): Number of hidden states
            random_state (int): Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.returns = None
        self.hidden_states = None
        self.is_fitted = False
        
    def prepare_data(self, returns):
        """
        Prepare returns data for HMM fitting
        
        Args:
            returns (pd.Series or np.array): Time series of returns
            
        Returns:
            np.array: Prepared data for HMM
        """
        if isinstance(returns, pd.Series):
            self.returns = returns.values
            self.dates = returns.index
        else:
            self.returns = returns
            self.dates = pd.date_range('2020-01-01', periods=len(returns), freq='D')
        
        # Reshape for hmmlearn (needs 2D array)
        X = self.returns.reshape(-1, 1)
        
        return X
    
    def fit(self, returns, algorithm="viterbi", n_iter=1000, tol=1e-4):
        """
        Fit Gaussian HMM to returns data
        
        Args:
            returns (pd.Series or np.array): Time series of returns
            algorithm (str): Algorithm for parameter estimation
            n_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
            
        Returns:
            self: Fitted model
        """
        print(f"Fitting Gaussian HMM with {self.n_states} states...")
        
        # Prepare data
        X = self.prepare_data(returns)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            algorithm=algorithm,
            n_iter=n_iter,
            tol=tol,
            random_state=self.random_state
        )
        
        try:
            self.model.fit(X)
            self.is_fitted = True
            
            # Decode hidden states
            self.hidden_states = self.model.predict(X)
            
            print("Model fitting completed successfully!")
            self._print_model_parameters()
            
        except Exception as e:
            print(f"Error fitting model: {e}")
            self.is_fitted = False
        
        return self
    
    def _print_model_parameters(self):
        """Print learned model parameters"""
        if not self.is_fitted:
            print("Model not fitted yet")
            return
        
        print("\nLearned Model Parameters:")
        print("-" * 40)
        
        print("Initial State Probabilities:")
        for i, prob in enumerate(self.model.startprob_):
            print(f"  State {i}: {prob:.4f}")
        
        print("\nTransition Matrix:")
        for i in range(self.n_states):
            row_str = "  "
            for j in range(self.n_states):
                row_str += f"{self.model.transmat_[i, j]:.4f}  "
            print(f"State {i}: {row_str}")
        
        print("\nEmission Parameters (Mean, Std):")
        for i in range(self.n_states):
            mean = self.model.means_[i, 0]
            std = np.sqrt(self.model.covars_[i, 0, 0])
            print(f"  State {i}: μ={mean:.6f}, σ={std:.6f}")
    
    def predict_states(self, returns=None):
        """
        Predict hidden states for given returns
        
        Args:
            returns (pd.Series or np.array): Returns data (uses training data if None)
            
        Returns:
            np.array: Predicted hidden states
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if returns is None:
            return self.hidden_states
        
        X = self.prepare_data(returns)
        return self.model.predict(X)
    
    def get_state_probabilities(self, returns=None):
        """
        Get state probabilities for each time point
        
        Args:
            returns (pd.Series or np.array): Returns data
            
        Returns:
            np.array: State probabilities (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if returns is None:
            X = self.returns.reshape(-1, 1)
        else:
            X = self.prepare_data(returns)
        
        return self.model.predict_proba(X)
    
    def calculate_log_likelihood(self, returns=None):
        """
        Calculate log-likelihood of the data
        
        Args:
            returns (pd.Series or np.array): Returns data
            
        Returns:
            float: Log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculation")
        
        if returns is None:
            X = self.returns.reshape(-1, 1)
        else:
            X = self.prepare_data(returns)
        
        return self.model.score(X)
    
    def analyze_states(self):
        """
        Analyze characteristics of each hidden state
        
        Returns:
            pd.DataFrame: State analysis results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analysis")
        
        state_analysis = []
        
        for state in range(self.n_states):
            # Get returns for this state
            state_mask = self.hidden_states == state
            state_returns = self.returns[state_mask]
            
            if len(state_returns) > 0:
                analysis = {
                    'State': state,
                    'Count': len(state_returns),
                    'Percentage': len(state_returns) / len(self.returns) * 100,
                    'Mean_Return': state_returns.mean(),
                    'Std_Return': state_returns.std(),
                    'Volatility_Annualized': state_returns.std() * np.sqrt(252),
                    'Min_Return': state_returns.min(),
                    'Max_Return': state_returns.max(),
                    'Skewness': stats.skew(state_returns),
                    'Kurtosis': stats.kurtosis(state_returns),
                    'VaR_5%': np.percentile(state_returns, 5),
                    'VaR_1%': np.percentile(state_returns, 1)
                }
            else:
                analysis = {
                    'State': state,
                    'Count': 0,
                    'Percentage': 0,
                    'Mean_Return': np.nan,
                    'Std_Return': np.nan,
                    'Volatility_Annualized': np.nan,
                    'Min_Return': np.nan,
                    'Max_Return': np.nan,
                    'Skewness': np.nan,
                    'Kurtosis': np.nan,
                    'VaR_5%': np.nan,
                    'VaR_1%': np.nan
                }
            
            state_analysis.append(analysis)
        
        return pd.DataFrame(state_analysis)
    
    def interpret_states(self):
        """
        Interpret states in financial context
        
        Returns:
            dict: State interpretations
        """
        state_df = self.analyze_states()
        interpretations = {}
        
        # Sort states by volatility for interpretation
        state_df_sorted = state_df.sort_values('Volatility_Annualized')
        
        regime_names = ['Low Volatility', 'Medium Volatility', 'High Volatility', 'Extreme Volatility']
        
        for i, (_, row) in enumerate(state_df_sorted.iterrows()):
            state_num = int(row['State'])
            vol = row['Volatility_Annualized']
            mean_ret = row['Mean_Return']
            
            # Determine regime type
            if i < len(regime_names):
                regime_type = regime_names[i]
            else:
                regime_type = f"Regime {i+1}"
            
            # Determine market condition
            if mean_ret > 0.001:  # Positive returns
                market_condition = "Bull Market"
            elif mean_ret < -0.001:  # Negative returns
                market_condition = "Bear Market"
            else:
                market_condition = "Neutral Market"
            
            interpretations[state_num] = {
                'regime_type': regime_type,
                'market_condition': market_condition,
                'volatility': vol,
                'mean_return': mean_ret,
                'description': self._get_state_description(vol, mean_ret)
            }
        
        return interpretations
    
    def _get_state_description(self, volatility, mean_return):
        """
        Get descriptive text for a state
        
        Args:
            volatility (float): Annualized volatility
            mean_return (float): Mean return
            
        Returns:
            str: State description
        """
        vol_desc = ""
        if volatility < 0.15:
            vol_desc = "low volatility"
        elif volatility < 0.25:
            vol_desc = "moderate volatility"
        elif volatility < 0.35:
            vol_desc = "high volatility"
        else:
            vol_desc = "extreme volatility"
        
        ret_desc = ""
        if mean_return > 0.002:
            ret_desc = "strong positive returns"
        elif mean_return > 0:
            ret_desc = "modest positive returns"
        elif mean_return > -0.002:
            ret_desc = "neutral returns"
        else:
            ret_desc = "negative returns"
        
        return f"Period of {vol_desc} with {ret_desc}"
    
    def predict_next_state(self, current_state=None, n_steps=1):
        """
        Predict next state(s) based on transition matrix
        
        Args:
            current_state (int): Current state (uses last observed state if None)
            n_steps (int): Number of steps ahead to predict
            
        Returns:
            dict: Prediction probabilities for each step
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if current_state is None:
            current_state = self.hidden_states[-1]
        
        predictions = {}
        
        for step in range(1, n_steps + 1):
            if step == 1:
                # Direct transition from current state
                probs = self.model.transmat_[current_state]
            else:
                # Multi-step transition
                transition_matrix_power = np.linalg.matrix_power(self.model.transmat_, step)
                probs = transition_matrix_power[current_state]
            
            predictions[f'step_{step}'] = {
                f'state_{i}': prob for i, prob in enumerate(probs)
            }
        
        return predictions
    
    def save_model(self, filepath):
        """
        Save the fitted model
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'n_states': self.n_states,
            'returns': self.returns,
            'dates': self.dates,
            'hidden_states': self.hidden_states,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            GaussianHMM: Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        hmm_model = cls(n_states=model_data['n_states'], 
                       random_state=model_data['random_state'])
        
        # Restore model state
        hmm_model.model = model_data['model']
        hmm_model.returns = model_data['returns']
        hmm_model.dates = model_data['dates']
        hmm_model.hidden_states = model_data['hidden_states']
        hmm_model.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        return hmm_model


class HMMModelSelector:
    """
    Class to help select optimal number of hidden states for HMM
    """
    
    def __init__(self, max_states=6, random_state=42):
        """
        Initialize model selector
        
        Args:
            max_states (int): Maximum number of states to test
            random_state (int): Random seed
        """
        self.max_states = max_states
        self.random_state = random_state
        self.results = None
    
    def select_best_model(self, returns, criteria=['aic', 'bic', 'log_likelihood']):
        """
        Select best number of states using various criteria
        
        Args:
            returns (pd.Series): Returns data
            criteria (list): Criteria to use for selection
            
        Returns:
            dict: Results for each number of states
        """
        print("Testing different numbers of hidden states...")
        
        results = {}
        
        for n_states in range(2, self.max_states + 1):
            print(f"Testing {n_states} states...")
            
            try:
                # Fit HMM
                hmm_model = GaussianHMM(n_states=n_states, random_state=self.random_state)
                hmm_model.fit(returns, n_iter=500)
                
                if hmm_model.is_fitted:
                    # Calculate information criteria
                    log_likelihood = hmm_model.calculate_log_likelihood()
                    n_params = n_states * (n_states - 1) + n_states * 2  # Simplified parameter count
                    n_samples = len(returns)
                    
                    aic = -2 * log_likelihood + 2 * n_params
                    bic = -2 * log_likelihood + np.log(n_samples) * n_params
                    
                    results[n_states] = {
                        'log_likelihood': log_likelihood,
                        'aic': aic,
                        'bic': bic,
                        'n_params': n_params,
                        'model': hmm_model
                    }
                    
                    print(f"  States: {n_states}, Log-likelihood: {log_likelihood:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}")
                
            except Exception as e:
                print(f"  Error with {n_states} states: {e}")
                continue
        
        self.results = results
        
        # Find best models by each criterion
        best_models = {}
        for criterion in criteria:
            if criterion in ['aic', 'bic']:
                best_n_states = min(results.keys(), key=lambda k: results[k][criterion])
            elif criterion == 'log_likelihood':
                best_n_states = max(results.keys(), key=lambda k: results[k][criterion])
            
            best_models[criterion] = {
                'n_states': best_n_states,
                'value': results[best_n_states][criterion],
                'model': results[best_n_states]['model']
            }
        
        print("\nBest models by criterion:")
        for criterion, info in best_models.items():
            print(f"  {criterion.upper()}: {info['n_states']} states (value: {info['value']:.2f})")
        
        return results, best_models
    
    def plot_model_selection(self, save_path=None):
        """
        Plot model selection criteria
        
        Args:
            save_path (str): Path to save plot
        """
        if self.results is None:
            raise ValueError("Must run model selection first")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        n_states = list(self.results.keys())
        aic_values = [self.results[n]['aic'] for n in n_states]
        bic_values = [self.results[n]['bic'] for n in n_states]
        ll_values = [self.results[n]['log_likelihood'] for n in n_states]
        
        # AIC plot
        axes[0].plot(n_states, aic_values, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of States')
        axes[0].set_ylabel('AIC')
        axes[0].set_title('AIC vs Number of States')
        axes[0].grid(True, alpha=0.3)
        min_aic_idx = np.argmin(aic_values)
        axes[0].plot(n_states[min_aic_idx], aic_values[min_aic_idx], 'ro', markersize=12, 
                    label=f'Best: {n_states[min_aic_idx]} states')
        axes[0].legend()
        
        # BIC plot
        axes[1].plot(n_states, bic_values, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of States')
        axes[1].set_ylabel('BIC')
        axes[1].set_title('BIC vs Number of States')
        axes[1].grid(True, alpha=0.3)
        min_bic_idx = np.argmin(bic_values)
        axes[1].plot(n_states[min_bic_idx], bic_values[min_bic_idx], 'ro', markersize=12,
                    label=f'Best: {n_states[min_bic_idx]} states')
        axes[1].legend()
        
        # Log-likelihood plot
        axes[2].plot(n_states, ll_values, 'mo-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Number of States')
        axes[2].set_ylabel('Log-likelihood')
        axes[2].set_title('Log-likelihood vs Number of States')
        axes[2].grid(True, alpha=0.3)
        max_ll_idx = np.argmax(ll_values)
        axes[2].plot(n_states[max_ll_idx], ll_values[max_ll_idx], 'ro', markersize=12,
                    label=f'Best: {n_states[max_ll_idx]} states')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model selection plot saved to {save_path}")
        
        plt.show()


def main():
    """
    Main function to demonstrate Gaussian HMM functionality
    """
    print("Gaussian Hidden Markov Model for Financial Analysis")
    print("=" * 55)
    
    # For demonstration, create synthetic returns data
    np.random.seed(42)
    
    # Simulate two-regime data
    n_samples = 1000
    regime_changes = [0, 400, 700, 1000]  # Change points
    returns = []
    
    for i in range(len(regime_changes) - 1):
        start, end = regime_changes[i], regime_changes[i + 1]
        n_points = end - start
        
        if i % 2 == 0:  # Low volatility regime
            regime_returns = np.random.normal(0.001, 0.01, n_points)
        else:  # High volatility regime
            regime_returns = np.random.normal(-0.002, 0.025, n_points)
        
        returns.extend(regime_returns)
    
    returns = np.array(returns)
    dates = pd.date_range('2020-01-01', periods=len(returns), freq='D')
    returns_series = pd.Series(returns, index=dates)
    
    print(f"Generated {len(returns)} synthetic return observations")
    
    # Test model selection
    selector = HMMModelSelector(max_states=5)
    results, best_models = selector.select_best_model(returns_series)
    
    # Fit best model (using BIC criterion)
    best_n_states = best_models['bic']['n_states']
    print(f"\nUsing {best_n_states} states based on BIC criterion")
    
    hmm_model = GaussianHMM(n_states=best_n_states)
    hmm_model.fit(returns_series)
    
    # Analyze states
    state_analysis = hmm_model.analyze_states()
    print("\nState Analysis:")
    print(state_analysis)
    
    # Get state interpretations
    interpretations = hmm_model.interpret_states()
    print("\nState Interpretations:")
    for state, info in interpretations.items():
        print(f"State {state}: {info['description']}")
    
    # Predict future states
    predictions = hmm_model.predict_next_state(n_steps=3)
    print("\nNext 3-step state predictions:")
    for step, probs in predictions.items():
        print(f"{step}: {probs}")
    
    print("\nGaussian HMM demonstration completed!")


if __name__ == "__main__":
    main()