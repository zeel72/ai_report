"""
Visualization and Analysis Module
Lab Assignment 5: Gaussian Hidden Markov Models for Financial Time Series Analysis
Course: CS307 - Artificial Intelligence
Week 5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

class HMMVisualizer:
    """
    Comprehensive visualization and analysis toolkit for Gaussian HMM results
    """
    
    def __init__(self, hmm_model, returns_data, ticker="Stock"):
        """
        Initialize visualizer with fitted HMM model and data
        
        Args:
            hmm_model: Fitted GaussianHMM object
            returns_data (pd.Series): Original returns data with dates
            ticker (str): Stock ticker for labeling
        """
        self.hmm_model = hmm_model
        self.returns_data = returns_data
        self.ticker = ticker
        self.state_colors = plt.cm.Set1(np.linspace(0, 1, hmm_model.n_states))
        
    def plot_returns_with_states(self, figsize=(15, 8), save_path=None):
        """
        Plot returns time series colored by hidden states
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Get state interpretations for legend
        interpretations = self.hmm_model.interpret_states()
        
        # Plot returns colored by states
        dates = self.returns_data.index
        returns = self.returns_data.values
        states = self.hmm_model.hidden_states
        
        for state in range(self.hmm_model.n_states):
            mask = states == state
            state_dates = dates[mask]
            state_returns = returns[mask]
            
            label = f"State {state}: {interpretations[state]['regime_type']}"
            ax1.scatter(state_dates, state_returns, 
                       c=[self.state_colors[state]], 
                       alpha=0.6, s=20, label=label)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title(f'{self.ticker} - Daily Returns Colored by Market Regime', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Daily Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot state sequence
        for state in range(self.hmm_model.n_states):
            mask = states == state
            ax2.fill_between(dates, 0, 1, where=mask, 
                           color=self.state_colors[state], alpha=0.7, 
                           step='pre', label=f'State {state}')
        
        ax2.set_title('Hidden State Sequence', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('State')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Returns with states plot saved to {save_path}")
        
        plt.show()
    
    def plot_price_with_regimes(self, price_data, figsize=(15, 10), save_path=None):
        """
        Plot stock price with regime overlays
        
        Args:
            price_data (pd.Series): Price data
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 2, 1])
        
        # Align price data with returns data (returns start from second day)
        aligned_price_data = price_data[1:].copy()  # Remove first day to match returns
        
        dates = aligned_price_data.index
        prices = aligned_price_data.values
        returns = self.returns_data.values
        states = self.hmm_model.hidden_states
        
        # Ensure all data has same length
        min_length = min(len(prices), len(returns), len(states))
        dates = dates[:min_length]
        prices = prices[:min_length] 
        returns = returns[:min_length]
        states = states[:min_length]
        
        interpretations = self.hmm_model.interpret_states()
        
        # Plot 1: Stock Price with regime backgrounds
        axes[0].plot(dates, prices, color='black', linewidth=1, alpha=0.8)
        
        # Add regime backgrounds
        for state in range(self.hmm_model.n_states):
            mask = states == state
            # Create continuous segments for each regime
            regime_periods = self._get_regime_periods(dates, mask)
            
            for start_date, end_date in regime_periods:
                axes[0].axvspan(start_date, end_date, 
                               color=self.state_colors[state], 
                               alpha=0.2, 
                               label=f'State {state}' if start_date == regime_periods[0][0] else "")
        
        axes[0].set_title(f'{self.ticker} - Price with Market Regimes', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Returns with volatility bands
        axes[1].plot(dates, returns, color='blue', linewidth=0.8, alpha=0.7)
        
        # Add volatility bands for each state
        state_analysis = self.hmm_model.analyze_states()
        for state in range(self.hmm_model.n_states):
            mask = states == state
            mean_ret = state_analysis.loc[state, 'Mean_Return']
            std_ret = state_analysis.loc[state, 'Std_Return']
            
            state_dates = dates[mask]
            if len(state_dates) > 0:
                axes[1].fill_between(state_dates, 
                                   mean_ret - 2*std_ret, 
                                   mean_ret + 2*std_ret,
                                   color=self.state_colors[state], 
                                   alpha=0.2)
        
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('Returns with Volatility Bands', fontsize=12)
        axes[1].set_ylabel('Daily Return')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: State probabilities over time
        state_probs = self.hmm_model.get_state_probabilities()
        
        for state in range(self.hmm_model.n_states):
            axes[2].plot(dates, state_probs[:, state], 
                        color=self.state_colors[state],
                        linewidth=2, alpha=0.8,
                        label=f'State {state}')
        
        axes[2].set_title('State Probabilities Over Time', fontsize=12)
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Probability')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Price with regimes plot saved to {save_path}")
        
        plt.show()
    
    def _get_regime_periods(self, dates, mask):
        """
        Helper function to get continuous periods for a regime
        
        Args:
            dates: Date index
            mask: Boolean mask for the regime
            
        Returns:
            list: List of (start_date, end_date) tuples
        """
        periods = []
        in_period = False
        start_date = None
        
        for i, (date, is_regime) in enumerate(zip(dates, mask)):
            if is_regime and not in_period:
                start_date = date
                in_period = True
            elif not is_regime and in_period:
                periods.append((start_date, date))
                in_period = False
        
        # Close the last period if needed
        if in_period:
            periods.append((start_date, dates[-1]))
        
        return periods
    
    def plot_transition_matrix(self, figsize=(8, 6), save_path=None):
        """
        Plot transition matrix as heatmap
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        transition_matrix = self.hmm_model.model.transmat_
        interpretations = self.hmm_model.interpret_states()
        
        # Create labels with regime types
        labels = [f"State {i}\n{interpretations[i]['regime_type']}" 
                 for i in range(self.hmm_model.n_states)]
        
        plt.figure(figsize=figsize)
        sns.heatmap(transition_matrix, 
                    annot=True, 
                    cmap='Blues', 
                    xticklabels=labels,
                    yticklabels=labels,
                    fmt='.3f',
                    cbar_kws={'label': 'Transition Probability'})
        
        plt.title('State Transition Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transition matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_state_distributions(self, figsize=(15, 10), save_path=None):
        """
        Plot return distributions for each state
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        n_states = self.hmm_model.n_states
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        returns = self.returns_data.values
        states = self.hmm_model.hidden_states
        interpretations = self.hmm_model.interpret_states()
        
        # Plot 1: Overlapped histograms
        for state in range(n_states):
            state_returns = returns[states == state]
            if len(state_returns) > 0:
                axes[0].hist(state_returns, bins=50, alpha=0.6, 
                           color=self.state_colors[state],
                           label=f'State {state}: {interpretations[state]["regime_type"]}',
                           density=True)
        
        axes[0].set_title('Return Distributions by State', fontweight='bold')
        axes[0].set_xlabel('Daily Return')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Box plots
        state_returns_list = []
        state_labels = []
        for state in range(n_states):
            state_returns = returns[states == state]
            if len(state_returns) > 0:
                state_returns_list.extend(state_returns)
                state_labels.extend([f'State {state}'] * len(state_returns))
        
        if state_returns_list:
            df_box = pd.DataFrame({'Returns': state_returns_list, 'State': state_labels})
            sns.boxplot(data=df_box, x='State', y='Returns', ax=axes[1])
            axes[1].set_title('Return Distributions (Box Plot)', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Volatility comparison
        state_analysis = self.hmm_model.analyze_states()
        volatilities = state_analysis['Volatility_Annualized'].values
        state_names = [f'State {i}' for i in range(n_states)]
        
        bars = axes[2].bar(state_names, volatilities, color=self.state_colors[:n_states])
        axes[2].set_title('Annualized Volatility by State', fontweight='bold')
        axes[2].set_ylabel('Volatility')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, vol in zip(bars, volatilities):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{vol:.1%}', ha='center', va='bottom')
        
        # Plot 4: Mean returns comparison
        mean_returns = state_analysis['Mean_Return'].values * 252  # Annualized
        bars = axes[3].bar(state_names, mean_returns, color=self.state_colors[:n_states])
        axes[3].set_title('Annualized Mean Return by State', fontweight='bold')
        axes[3].set_ylabel('Annualized Return')
        axes[3].grid(True, alpha=0.3, axis='y')
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, ret in zip(bars, mean_returns):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height,
                        f'{ret:.1%}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"State distributions plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_plot(self, price_data, save_path=None):
        """
        Create interactive plot using Plotly
        
        Args:
            price_data (pd.Series): Price data
            save_path (str): Path to save HTML file
        """
        # Create subplot figure
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=['Stock Price with Market Regimes', 
                                         'Daily Returns', 
                                         'State Probabilities'],
                           row_heights=[0.5, 0.3, 0.2])
        
        # Align price data with returns data
        aligned_price_data = price_data[1:].copy()  # Remove first day to match returns
        
        dates = aligned_price_data.index
        prices = aligned_price_data.values
        returns = self.returns_data.values
        states = self.hmm_model.hidden_states
        state_probs = self.hmm_model.get_state_probabilities()
        
        # Ensure all data has same length
        min_length = min(len(prices), len(returns), len(states), len(state_probs))
        dates = dates[:min_length]
        prices = prices[:min_length]
        returns = returns[:min_length]
        states = states[:min_length]
        state_probs = state_probs[:min_length]
        
        interpretations = self.hmm_model.interpret_states()
        colors = px.colors.qualitative.Set1[:self.hmm_model.n_states]
        
        # Add price line
        fig.add_trace(
            go.Scatter(x=dates, y=prices, mode='lines', name='Price',
                      line=dict(color='black', width=1)),
            row=1, col=1
        )
        
        # Add regime backgrounds to price chart
        for state in range(self.hmm_model.n_states):
            mask = states == state
            regime_periods = self._get_regime_periods(dates, mask)
            
            for i, (start_date, end_date) in enumerate(regime_periods):
                fig.add_vrect(
                    x0=start_date, x1=end_date,
                    fillcolor=colors[state], opacity=0.2,
                    layer="below", line_width=0,
                    row=1, col=1
                )
                
                # Add legend only for first occurrence
                if i == 0:
                    fig.add_trace(
                        go.Scatter(x=[start_date], y=[prices[0]], 
                                  mode='markers', opacity=0,
                                  name=f'State {state}: {interpretations[state]["regime_type"]}',
                                  marker=dict(color=colors[state])),
                        row=1, col=1
                    )
        
        # Add returns
        fig.add_trace(
            go.Scatter(x=dates, y=returns, mode='lines', name='Returns',
                      line=dict(color='blue', width=0.8)),
            row=2, col=1
        )
        
        # Add zero line for returns
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        # Add state probabilities
        for state in range(self.hmm_model.n_states):
            fig.add_trace(
                go.Scatter(x=dates, y=state_probs[:, state], 
                          mode='lines', 
                          name=f'State {state} Prob',
                          line=dict(color=colors[state], width=2)),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{self.ticker} - Hidden Markov Model Analysis',
            height=800,
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        
        fig.show()
    
    def generate_analysis_report(self):
        """
        Generate comprehensive analysis report
        
        Returns:
            dict: Analysis report
        """
        state_analysis = self.hmm_model.analyze_states()
        interpretations = self.hmm_model.interpret_states()
        transition_matrix = self.hmm_model.model.transmat_
        
        report = {
            'model_info': {
                'n_states': self.hmm_model.n_states,
                'ticker': self.ticker,
                'data_points': len(self.returns_data),
                'date_range': f"{self.returns_data.index[0].strftime('%Y-%m-%d')} to {self.returns_data.index[-1].strftime('%Y-%m-%d')}"
            },
            'state_analysis': state_analysis.to_dict('records'),
            'state_interpretations': interpretations,
            'transition_analysis': self._analyze_transitions(transition_matrix, interpretations),
            'market_insights': self._generate_market_insights(state_analysis, interpretations),
            'risk_analysis': self._generate_risk_analysis(state_analysis)
        }
        
        return report
    
    def _analyze_transitions(self, transition_matrix, interpretations):
        """
        Analyze transition patterns
        
        Args:
            transition_matrix (np.array): Transition probabilities
            interpretations (dict): State interpretations
            
        Returns:
            dict: Transition analysis
        """
        n_states = len(transition_matrix)
        
        # Find most persistent states
        persistence = [transition_matrix[i, i] for i in range(n_states)]
        most_persistent = np.argmax(persistence)
        
        # Find most volatile transitions
        off_diagonal = []
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    off_diagonal.append({
                        'from_state': i,
                        'to_state': j,
                        'probability': transition_matrix[i, j],
                        'from_regime': interpretations[i]['regime_type'],
                        'to_regime': interpretations[j]['regime_type']
                    })
        
        # Sort by probability
        off_diagonal.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'most_persistent_state': {
                'state': most_persistent,
                'persistence': persistence[most_persistent],
                'regime': interpretations[most_persistent]['regime_type']
            },
            'persistence_by_state': [
                {
                    'state': i,
                    'persistence': persistence[i],
                    'regime': interpretations[i]['regime_type']
                }
                for i in range(n_states)
            ],
            'top_transitions': off_diagonal[:5]  # Top 5 transitions
        }
    
    def _generate_market_insights(self, state_analysis, interpretations):
        """
        Generate market insights from the analysis
        
        Args:
            state_analysis (pd.DataFrame): State analysis results
            interpretations (dict): State interpretations
            
        Returns:
            list: List of insights
        """
        insights = []
        
        # Find high volatility periods
        high_vol_states = state_analysis[state_analysis['Volatility_Annualized'] > 0.3]
        if not high_vol_states.empty:
            for _, state in high_vol_states.iterrows():
                insights.append(
                    f"State {int(state['State'])} represents a high-risk period with "
                    f"{state['Volatility_Annualized']:.1%} annualized volatility, "
                    f"occurring {state['Percentage']:.1f}% of the time."
                )
        
        # Find best performing periods
        best_return_state = state_analysis.loc[state_analysis['Mean_Return'].idxmax()]
        insights.append(
            f"The most profitable regime is State {int(best_return_state['State'])} "
            f"({interpretations[int(best_return_state['State'])]['regime_type']}) with "
            f"{best_return_state['Mean_Return']*252:.1%} annualized returns."
        )
        
        # Find most common regime
        most_common_state = state_analysis.loc[state_analysis['Percentage'].idxmax()]
        insights.append(
            f"The market spends most time ({most_common_state['Percentage']:.1f}%) "
            f"in State {int(most_common_state['State'])} "
            f"({interpretations[int(most_common_state['State'])]['regime_type']})."
        )
        
        return insights
    
    def _generate_risk_analysis(self, state_analysis):
        """
        Generate risk analysis
        
        Args:
            state_analysis (pd.DataFrame): State analysis results
            
        Returns:
            dict: Risk analysis
        """
        # Calculate Value at Risk for each state
        var_analysis = []
        for _, state in state_analysis.iterrows():
            if not pd.isna(state['VaR_5%']):
                var_analysis.append({
                    'state': int(state['State']),
                    'var_5%': state['VaR_5%'],
                    'var_1%': state['VaR_1%'],
                    'volatility': state['Volatility_Annualized'],
                    'max_drawdown': state['Min_Return']
                })
        
        # Overall portfolio risk metrics
        all_returns = self.returns_data.values
        portfolio_var_5 = np.percentile(all_returns, 5)
        portfolio_var_1 = np.percentile(all_returns, 1)
        
        return {
            'state_var_analysis': var_analysis,
            'portfolio_metrics': {
                'overall_var_5%': portfolio_var_5,
                'overall_var_1%': portfolio_var_1,
                'overall_volatility': np.std(all_returns) * np.sqrt(252),
                'sharpe_ratio': np.mean(all_returns) / np.std(all_returns) * np.sqrt(252)
            }
        }
    
    def save_all_plots(self, output_dir, price_data=None):
        """
        Save all plots to specified directory
        
        Args:
            output_dir (str): Output directory
            price_data (pd.Series): Price data (optional)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving all plots to {output_dir}...")
        
        # Returns with states
        self.plot_returns_with_states(save_path=f"{output_dir}/returns_with_states.png")
        
        # Price with regimes (if price data available)
        if price_data is not None:
            self.plot_price_with_regimes(price_data, save_path=f"{output_dir}/price_with_regimes.png")
            self.create_interactive_plot(price_data, save_path=f"{output_dir}/interactive_analysis.html")
        
        # Transition matrix
        self.plot_transition_matrix(save_path=f"{output_dir}/transition_matrix.png")
        
        # State distributions
        self.plot_state_distributions(save_path=f"{output_dir}/state_distributions.png")
        
        print("All plots saved successfully!")


def main():
    """
    Main function to demonstrate visualization functionality
    """
    print("HMM Visualization and Analysis Demo")
    print("=" * 40)
    
    # This would typically be called with real HMM results
    print("This module provides comprehensive visualization tools for HMM analysis.")
    print("Use it in conjunction with the GaussianHMM class for full functionality.")


if __name__ == "__main__":
    main()