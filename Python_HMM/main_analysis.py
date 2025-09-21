"""
Complete Financial Time Series Analysis using Gaussian Hidden Markov Models
Lab Assignment 5: Gaussian Hidden Markov Models for Financial Time Series Analysis
Course: CS307 - Artificial Intelligence
Week 5

This is the main script that demonstrates the complete workflow for financial analysis using HMM.
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from data_collection import FinancialDataCollector
from gaussian_hmm import GaussianHMM, HMMModelSelector
from visualization import HMMVisualizer

warnings.filterwarnings('ignore')

def main_analysis():
    """
    Complete financial time series analysis workflow
    """
    print("="*80)
    print("FINANCIAL TIME SERIES ANALYSIS USING GAUSSIAN HIDDEN MARKOV MODELS")
    print("Lab Assignment 5 - CS307 Artificial Intelligence")
    print("="*80)
    
    # Configuration
    TICKER = "AAPL"  # Can be changed to TSLA, ^GSPC, etc.
    PERIOD = "10y"   # 10 years of data
    OUTPUT_DIR = "../results"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    
    # =========================================================================
    # PART 1: DATA COLLECTION AND PREPROCESSING
    # =========================================================================
    print("\nPART 1: DATA COLLECTION AND PREPROCESSING")
    print("-" * 50)
    
    try:
        # Initialize data collector
        collector = FinancialDataCollector()
        
        # Download data
        print(f"Downloading {TICKER} data for {PERIOD}...")
        stock_data = collector.download_data(TICKER, period=PERIOD)
        
        if stock_data is None:
            print("Error: Could not download data. Please check ticker symbol and internet connection.")
            return
        
        # Preprocess data
        print("Preprocessing data...")
        returns = collector.preprocess_data()
        
        # Clean data
        cleaned_returns = collector.clean_data()
        
        # Save processed data
        collector.save_data()
        
        # Create overview plots
        print("Creating data overview plots...")
        collector.plot_data_overview(f"{OUTPUT_DIR}/plots/data_overview.png")
        
        print(f"✓ Data collection completed: {len(returns)} trading days")
        
    except Exception as e:
        print(f"Error in data collection: {e}")
        return
    
    # =========================================================================
    # PART 2: MODEL SELECTION AND FITTING
    # =========================================================================
    print("\nPART 2: GAUSSIAN HIDDEN MARKOV MODEL")
    print("-" * 50)
    
    try:
        # Model Selection
        print("Selecting optimal number of hidden states...")
        selector = HMMModelSelector(max_states=5, random_state=42)
        results, best_models = selector.select_best_model(cleaned_returns)
        
        # Plot model selection results
        selector.plot_model_selection(f"{OUTPUT_DIR}/plots/model_selection.png")
        
        # Use BIC criterion for final model
        best_n_states = best_models['bic']['n_states']
        print(f"✓ Selected {best_n_states} states based on BIC criterion")
        
        # Fit final model
        print(f"Fitting Gaussian HMM with {best_n_states} states...")
        hmm_model = GaussianHMM(n_states=best_n_states, random_state=42)
        hmm_model.fit(cleaned_returns, n_iter=1000)
        
        if not hmm_model.is_fitted:
            print("Error: Model fitting failed")
            return
        
        # Save model
        hmm_model.save_model(f"{OUTPUT_DIR}/hmm_model.pkl")
        print("✓ Model fitted and saved successfully")
        
    except Exception as e:
        print(f"Error in model fitting: {e}")
        return
    
    # =========================================================================
    # PART 3: STATE ANALYSIS AND INTERPRETATION
    # =========================================================================
    print("\nPART 3: INTERPRETATION AND INFERENCE")
    print("-" * 50)
    
    try:
        # Analyze states
        print("Analyzing hidden states...")
        state_analysis = hmm_model.analyze_states()
        print("\nState Analysis Summary:")
        print(state_analysis.round(4))
        
        # Get state interpretations
        interpretations = hmm_model.interpret_states()
        print("\nState Interpretations:")
        for state, info in interpretations.items():
            print(f"  State {state}: {info['description']}")
            print(f"    - Regime: {info['regime_type']}")
            print(f"    - Market: {info['market_condition']}")
            print(f"    - Volatility: {info['volatility']:.2%}")
        
        # Transition matrix analysis
        print("\nTransition Matrix Analysis:")
        transition_matrix = hmm_model.model.transmat_
        print("Transition probabilities:")
        for i in range(best_n_states):
            for j in range(best_n_states):
                if i != j:
                    prob = transition_matrix[i, j]
                    if prob > 0.1:  # Only show significant transitions
                        from_regime = interpretations[i]['regime_type']
                        to_regime = interpretations[j]['regime_type']
                        print(f"  {from_regime} → {to_regime}: {prob:.3f}")
        
        print("✓ State analysis completed")
        
    except Exception as e:
        print(f"Error in state analysis: {e}")
        return
    
    # =========================================================================
    # PART 4: VISUALIZATION
    # =========================================================================
    print("\nPART 4: EVALUATION AND VISUALIZATION")
    print("-" * 50)
    
    try:
        # Initialize visualizer
        visualizer = HMMVisualizer(hmm_model, cleaned_returns, TICKER)
        
        # Create all visualizations
        print("Creating visualizations...")
        
        # Returns with states
        visualizer.plot_returns_with_states(
            save_path=f"{OUTPUT_DIR}/plots/returns_with_states.png"
        )
        
        # Price with regimes - use available price column
        price_column = 'Close' if 'Close' in stock_data.columns else 'Adj Close'
        price_data = stock_data[price_column]
        visualizer.plot_price_with_regimes(
            price_data, 
            save_path=f"{OUTPUT_DIR}/plots/price_with_regimes.png"
        )
        
        # Transition matrix
        visualizer.plot_transition_matrix(
            save_path=f"{OUTPUT_DIR}/plots/transition_matrix.png"
        )
        
        # State distributions
        visualizer.plot_state_distributions(
            save_path=f"{OUTPUT_DIR}/plots/state_distributions.png"
        )
        
        # Interactive plot
        visualizer.create_interactive_plot(
            price_data,
            save_path=f"{OUTPUT_DIR}/plots/interactive_analysis.html"
        )
        
        print("✓ All visualizations created")
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        return
    
    # =========================================================================
    # PART 5: CONCLUSIONS AND INSIGHTS
    # =========================================================================
    print("\nPART 5: CONCLUSIONS AND INSIGHTS")
    print("-" * 50)
    
    try:
        # Generate comprehensive report
        report = visualizer.generate_analysis_report()
        
        # Save report as JSON
        with open(f"{OUTPUT_DIR}/analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display key insights
        print("\nKEY MARKET INSIGHTS:")
        for insight in report['market_insights']:
            print(f"• {insight}")
        
        # Display risk analysis
        print("\nRISK ANALYSIS:")
        risk_analysis = report['risk_analysis']
        portfolio_metrics = risk_analysis['portfolio_metrics']
        
        print(f"• Overall Volatility: {portfolio_metrics['overall_volatility']:.2%}")
        print(f"• 5% Value at Risk: {portfolio_metrics['overall_var_5%']:.4f}")
        print(f"• 1% Value at Risk: {portfolio_metrics['overall_var_1%']:.4f}")
        print(f"• Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
        
        # Future predictions
        print("\nFUTURE STATE PREDICTIONS:")
        current_state = hmm_model.hidden_states[-1]
        predictions = hmm_model.predict_next_state(current_state, n_steps=5)
        
        print(f"Current state: {current_state} ({interpretations[current_state]['regime_type']})")
        for step, probs in predictions.items():
            most_likely_state = max(probs, key=probs.get)
            most_likely_prob = probs[most_likely_state]
            state_num = int(most_likely_state.split('_')[1])
            regime = interpretations[state_num]['regime_type']
            print(f"  {step.replace('_', ' ').title()}: {regime} (probability: {most_likely_prob:.3f})")
        
        print("✓ Analysis completed successfully")
        
    except Exception as e:
        print(f"Error in report generation: {e}")
        return
    
    # =========================================================================
    # BONUS: MULTI-ASSET COMPARISON
    # =========================================================================
    print("\nBONUS: MULTI-ASSET COMPARISON")
    print("-" * 50)
    
    try:
        # Compare multiple assets
        tickers = ['AAPL', 'TSLA', '^GSPC', 'GOOGL']
        comparison_results = {}
        
        print("Analyzing multiple assets for comparison...")
        
        for ticker in tickers:
            if ticker == TICKER:
                # Use already fitted model
                comparison_results[ticker] = {
                    'model': hmm_model,
                    'returns': cleaned_returns,
                    'interpretations': interpretations
                }
            else:
                try:
                    # Fit model for other tickers
                    temp_collector = FinancialDataCollector()
                    temp_data = temp_collector.download_data(ticker, period="5y")  # Shorter period for comparison
                    
                    if temp_data is not None:
                        temp_returns = temp_collector.preprocess_data()
                        temp_cleaned = temp_collector.clean_data()
                        
                        temp_hmm = GaussianHMM(n_states=2, random_state=42)  # Use 2 states for comparison
                        temp_hmm.fit(temp_cleaned)
                        
                        if temp_hmm.is_fitted:
                            comparison_results[ticker] = {
                                'model': temp_hmm,
                                'returns': temp_cleaned,
                                'interpretations': temp_hmm.interpret_states()
                            }
                            print(f"  ✓ {ticker} analysis completed")
                        
                except Exception as e:
                    print(f"  ✗ Error analyzing {ticker}: {e}")
                    continue
        
        # Create comparison summary
        if len(comparison_results) > 1:
            print(f"\nComparison Summary ({len(comparison_results)} assets):")
            for ticker, data in comparison_results.items():
                state_analysis = data['model'].analyze_states()
                max_vol = state_analysis['Volatility_Annualized'].max()
                print(f"  {ticker}: Max volatility regime = {max_vol:.2%}")
        
    except Exception as e:
        print(f"Error in multi-asset comparison: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("Files created:")
    print(f"  • Data files: {OUTPUT_DIR}/AAPL_*.csv")
    print(f"  • Model: {OUTPUT_DIR}/hmm_model.pkl")
    print(f"  • Report: {OUTPUT_DIR}/analysis_report.json")
    print(f"  • Plots: {OUTPUT_DIR}/plots/")
    print("    - data_overview.png")
    print("    - model_selection.png") 
    print("    - returns_with_states.png")
    print("    - price_with_regimes.png")
    print("    - transition_matrix.png")
    print("    - state_distributions.png")
    print("    - interactive_analysis.html")
    print("\nOpen interactive_analysis.html in a web browser for interactive exploration!")


def analyze_custom_ticker(ticker, period="10y", n_states=None):
    """
    Analyze a custom ticker with specified parameters
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for analysis
        n_states (int): Number of states (auto-select if None)
    """
    print(f"Analyzing {ticker} for period {period}")
    print("-" * 50)
    
    # Data collection
    collector = FinancialDataCollector()
    data = collector.download_data(ticker, period=period)
    
    if data is None:
        print(f"Error: Could not download data for {ticker}")
        return
    
    returns = collector.preprocess_data()
    cleaned_returns = collector.clean_data()
    
    # Model fitting
    if n_states is None:
        # Auto-select number of states
        selector = HMMModelSelector(max_states=4)
        results, best_models = selector.select_best_model(cleaned_returns)
        n_states = best_models['bic']['n_states']
    
    hmm_model = GaussianHMM(n_states=n_states)
    hmm_model.fit(cleaned_returns)
    
    if hmm_model.is_fitted:
        # Analysis
        state_analysis = hmm_model.analyze_states()
        interpretations = hmm_model.interpret_states()
        
        print(f"\nAnalysis Results for {ticker}:")
        print(f"Number of states: {n_states}")
        print("State interpretations:")
        for state, info in interpretations.items():
            print(f"  State {state}: {info['description']}")
        
        return hmm_model, cleaned_returns, data
    
    return None, None, None


if __name__ == "__main__":
    # Check if custom ticker provided as command line argument
    if len(sys.argv) > 1:
        custom_ticker = sys.argv[1].upper()
        period = sys.argv[2] if len(sys.argv) > 2 else "10y"
        
        print(f"Running custom analysis for {custom_ticker}")
        analyze_custom_ticker(custom_ticker, period)
    else:
        # Run main analysis
        main_analysis()