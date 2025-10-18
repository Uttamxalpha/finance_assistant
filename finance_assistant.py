"""
Personal Finance Assistant
Predicts monthly expenses and offers budgeting suggestions using ML

Requirements:
pip install pandas numpy scikit-learn matplotlib seaborn

Features:
- Transaction categorization
- Expense prediction using ML
- Budget recommendations
- Spending pattern analysis
- Anomaly detection
- Financial health scoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import os
import json

warnings.filterwarnings('ignore')

class PersonalFinanceAssistant:
    def __init__(self, data_file='transactions.csv'):
        """Initialize the finance assistant"""
        self.data_file = data_file
        self.transactions = None
        self.prediction_model = None
        self.label_encoders = {}
        self.categories = [
            'Groceries', 'Restaurants', 'Transportation', 'Entertainment',
            'Shopping', 'Bills', 'Healthcare', 'Education', 'Travel',
            'Investments', 'Other', 'Income'
        ]
        
        print("Personal Finance Assistant Initialized")
    
    def load_or_generate_data(self):
        """Load existing data or generate sample data"""
        if os.path.exists(self.data_file):
            print(f"Loading transactions from {self.data_file}")
            self.transactions = pd.read_csv(self.data_file)
            self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        else:
            print("Generating sample transaction data...")
            self.transactions = self._generate_sample_data()
            self.transactions.to_csv(self.data_file, index=False)
            print(f"Sample data saved to {self.data_file}")
        
        print(f"Loaded {len(self.transactions)} transactions")
        return self.transactions
    
    def _generate_sample_data(self, months=12):
        """Generate realistic sample transaction data"""
        np.random.seed(42)
        
        start_date = datetime.now() - timedelta(days=months*30)
        transactions = []
        
        # Monthly patterns
        monthly_patterns = {
            'Groceries': (300, 500, 20),  # (min, max, frequency per month)
            'Restaurants': (150, 300, 15),
            'Transportation': (100, 200, 25),
            'Entertainment': (50, 150, 8),
            'Shopping': (100, 400, 10),
            'Bills': (800, 1200, 5),
            'Healthcare': (50, 300, 3),
            'Education': (0, 500, 2),
            'Travel': (0, 1000, 1),
            'Investments': (200, 500, 1),
            'Income': (3000, 5000, 2)
        }
        
        current_date = start_date
        
        for month in range(months):
            for category, (min_amt, max_amt, freq) in monthly_patterns.items():
                for _ in range(freq):
                    day_offset = np.random.randint(0, 30)
                    transaction_date = current_date + timedelta(days=day_offset)
                    
                    if category == 'Income':
                        amount = np.random.uniform(min_amt, max_amt)
                        transaction_type = 'Income'
                    else:
                        amount = -np.random.uniform(min_amt, max_amt)
                        transaction_type = 'Expense'
                    
                    # Add some randomness
                    if np.random.random() < 0.1:  # 10% chance of skipping
                        continue
                    
                    transactions.append({
                        'date': transaction_date.strftime('%Y-%m-%d'),
                        'amount': round(amount, 2),
                        'category': category,
                        'type': transaction_type,
                        'description': self._generate_description(category)
                    })
            
            current_date += timedelta(days=30)
        
        df = pd.DataFrame(transactions)
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def _generate_description(self, category):
        """Generate realistic transaction descriptions"""
        descriptions = {
            'Groceries': ['Walmart', 'Target', 'Whole Foods', 'Local Market'],
            'Restaurants': ['Starbucks', 'McDonalds', 'Local Cafe', 'Pizza Place'],
            'Transportation': ['Uber', 'Gas Station', 'Parking', 'Public Transit'],
            'Entertainment': ['Netflix', 'Cinema', 'Concert', 'Spotify'],
            'Shopping': ['Amazon', 'Mall', 'Online Store', 'Boutique'],
            'Bills': ['Rent', 'Electricity', 'Internet', 'Phone'],
            'Healthcare': ['Pharmacy', 'Doctor Visit', 'Insurance', 'Gym'],
            'Education': ['Books', 'Course', 'Tuition', 'Supplies'],
            'Travel': ['Flight', 'Hotel', 'Vacation', 'Trip'],
            'Investments': ['Stock Purchase', '401k', 'Savings', 'Crypto'],
            'Income': ['Salary', 'Freelance', 'Bonus', 'Investment Return']
        }
        return np.random.choice(descriptions.get(category, ['Transaction']))
    
    def analyze_spending_patterns(self):
        """Analyze spending patterns and trends"""
        print("\n" + "="*70)
        print("SPENDING PATTERN ANALYSIS")
        print("="*70)
        
        # Calculate monthly statistics
        self.transactions['month'] = pd.to_datetime(self.transactions['date']).dt.to_period('M')
        expenses = self.transactions[self.transactions['type'] == 'Expense'].copy()
        income = self.transactions[self.transactions['type'] == 'Income'].copy()
        
        # Monthly totals
        monthly_expenses = expenses.groupby('month')['amount'].sum().abs()
        monthly_income = income.groupby('month')['amount'].sum()
        
        print(f"\nAverage Monthly Income: ${monthly_income.mean():.2f}")
        print(f"Average Monthly Expenses: ${monthly_expenses.mean():.2f}")
        print(f"Average Monthly Savings: ${(monthly_income.mean() - monthly_expenses.mean()):.2f}")
        
        # Category breakdown
        print(f"\nSpending by Category (Monthly Average):")
        category_avg = expenses.groupby('category')['amount'].sum().abs() / len(monthly_expenses)
        category_avg_sorted = category_avg.sort_values(ascending=False)
        
        for cat, amt in category_avg_sorted.items():
            percentage = (amt / monthly_expenses.mean()) * 100
            print(f"  {cat:.<20} ${amt:>8.2f}  ({percentage:>5.1f}%)")
        
        # Identify trends
        print(f"\nSpending Trends:")
        recent_3_months = monthly_expenses.tail(3).mean()
        previous_3_months = monthly_expenses.head(len(monthly_expenses)-3).tail(3).mean()
        
        trend_change = ((recent_3_months - previous_3_months) / previous_3_months) * 100
        
        if trend_change > 5:
            print(f"  WARNING: Spending increased by {trend_change:.1f}% in recent months")
        elif trend_change < -5:
            print(f"  GOOD: Spending decreased by {abs(trend_change):.1f}% in recent months")
        else:
            print(f"  Spending remained stable ({trend_change:+.1f}%)")
        
        return monthly_expenses, monthly_income, category_avg_sorted
    
    def train_prediction_model(self):
        """Train ML model to predict future expenses"""
        print("\n" + "="*70)
        print("TRAINING EXPENSE PREDICTION MODEL")
        print("="*70)
        
        # Prepare features
        expenses = self.transactions[self.transactions['type'] == 'Expense'].copy()
        expenses['month'] = pd.to_datetime(expenses['date']).dt.month
        expenses['day'] = pd.to_datetime(expenses['date']).dt.day
        expenses['day_of_week'] = pd.to_datetime(expenses['date']).dt.dayofweek
        
        # Aggregate by month and category
        monthly_data = expenses.groupby([
            pd.to_datetime(expenses['date']).dt.to_period('M'),
            'category'
        ])['amount'].sum().abs().reset_index()
        
        monthly_data.columns = ['month', 'category', 'amount']
        monthly_data['month_num'] = monthly_data['month'].apply(lambda x: x.month)
        
        # Encode categories
        le = LabelEncoder()
        monthly_data['category_encoded'] = le.fit_transform(monthly_data['category'])
        self.label_encoders['category'] = le
        
        # Create features
        X = monthly_data[['month_num', 'category_encoded']]
        y = monthly_data['amount']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.prediction_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.prediction_model.fit(X_train, y_train)
        
        # Evaluate
        score = self.prediction_model.score(X_test, y_test)
        print(f"\nModel trained successfully!")
        print(f"  R2 Score: {score:.3f}")
        print(f"  Prediction accuracy: ~{score*100:.1f}%")
        
        return self.prediction_model
    
    def predict_expenses(self, target_month=None):
        """Predict expenses for a specific month"""
        if self.prediction_model is None:
            print("Model not trained yet. Training now...")
            self.train_prediction_model()
        
        if target_month is None:
            # Get user input
            print("\n" + "="*70)
            print("EXPENSE PREDICTION")
            print("="*70)
            print("\nEnter the month number (1-12) to predict expenses:")
            print("Example: 1 for January, 12 for December")
            
            while True:
                try:
                    target_month = int(input("Month (1-12): "))
                    if 1 <= target_month <= 12:
                        break
                    else:
                        print("Please enter a number between 1 and 12")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        print(f"\nPredicting expenses for {month_names[target_month]}...")
        
        predictions = {}
        total_predicted = 0
        
        for category in self.categories:
            if category == 'Income':
                continue
            
            try:
                category_encoded = self.label_encoders['category'].transform([category])[0]
                features = [[target_month, category_encoded]]
                predicted_amount = self.prediction_model.predict(features)[0]
                predictions[category] = max(0, predicted_amount)
                total_predicted += predictions[category]
            except:
                predictions[category] = 0
        
        print(f"\nPredicted expenses for {month_names[target_month]}:")
        print(f"\n{'Category':<20} {'Predicted Amount':>15}")
        print("-" * 37)
        
        for cat, amt in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            if amt > 0:
                print(f"{cat:<20} ${amt:>14.2f}")
        
        print("-" * 37)
        print(f"{'TOTAL':<20} ${total_predicted:>14.2f}")
        
        return predictions, total_predicted
    
    def predict_custom_scenario(self):
        """Allow user to input custom spending scenarios"""
        print("\n" + "="*70)
        print("CUSTOM SCENARIO PREDICTION")
        print("="*70)
        print("\nWould you like to predict expenses for multiple months? (yes/no)")
        
        response = input("> ").strip().lower()
        
        if response in ['yes', 'y']:
            print("\nHow many months would you like to predict? (1-12)")
            try:
                num_months = int(input("> "))
                num_months = min(max(1, num_months), 12)
                
                current_month = datetime.now().month
                total_predictions = {}
                
                for i in range(num_months):
                    month = ((current_month + i - 1) % 12) + 1
                    predictions, total = self.predict_expenses(month)
                    total_predictions[month] = total
                
                print("\n" + "="*70)
                print("MULTI-MONTH PREDICTION SUMMARY")
                print("="*70)
                
                month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                
                for month, total in total_predictions.items():
                    print(f"{month_names[month]:<15} ${total:>10.2f}")
                
                print("-" * 37)
                print(f"{'Total':<15} ${sum(total_predictions.values()):>10.2f}")
                print(f"{'Average':<15} ${sum(total_predictions.values())/len(total_predictions):>10.2f}")
                
            except ValueError:
                print("Invalid input. Returning to menu.")
        else:
            self.predict_expenses()
    
    def detect_anomalies(self):
        """Detect unusual transactions"""
        print("\n" + "="*70)
        print("ANOMALY DETECTION")
        print("="*70)
        
        expenses = self.transactions[self.transactions['type'] == 'Expense'].copy()
        
        # Prepare features for anomaly detection
        expenses['month'] = pd.to_datetime(expenses['date']).dt.month
        expenses['day'] = pd.to_datetime(expenses['date']).dt.day
        
        le = LabelEncoder()
        expenses['category_encoded'] = le.fit_transform(expenses['category'])
        
        X = expenses[['amount', 'month', 'day', 'category_encoded']].abs()
        
        # Train isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        expenses['anomaly'] = iso_forest.fit_predict(X)
        
        # Get anomalies
        anomalies = expenses[expenses['anomaly'] == -1].copy()
        
        if len(anomalies) > 0:
            print(f"\nFound {len(anomalies)} unusual transactions:\n")
            
            for idx, row in anomalies.tail(10).iterrows():
                print(f"  {row['date']}: {row['description']} ({row['category']})")
                print(f"    Amount: ${abs(row['amount']):.2f}")
                print()
        else:
            print("\nNo unusual transactions detected")
        
        return anomalies
    
    def generate_budget_recommendations(self):
        """Generate personalized budget recommendations"""
        print("\n" + "="*70)
        print("BUDGET RECOMMENDATIONS")
        print("="*70)
        
        # Calculate current spending
        expenses = self.transactions[self.transactions['type'] == 'Expense'].copy()
        income = self.transactions[self.transactions['type'] == 'Income'].copy()
        
        # Ensure date is datetime
        expenses['date'] = pd.to_datetime(expenses['date'])
        income['date'] = pd.to_datetime(income['date'])
        
        monthly_income = income.groupby(income['date'].dt.to_period('M'))['amount'].sum().mean()
        monthly_expenses = expenses.groupby(expenses['date'].dt.to_period('M'))['amount'].sum().abs().mean()
        
        category_spending = expenses.groupby('category')['amount'].sum().abs()
        num_months = len(expenses['date'].dt.to_period('M').unique())
        category_avg = category_spending / num_months
        
        # 50/30/20 rule: 50% needs, 30% wants, 20% savings
        recommended_savings = monthly_income * 0.20
        recommended_needs = monthly_income * 0.50
        recommended_wants = monthly_income * 0.30
        
        current_savings = monthly_income - monthly_expenses
        savings_rate = (current_savings / monthly_income) * 100
        
        print(f"\nCurrent Financial Overview:")
        print(f"  Monthly Income:        ${monthly_income:.2f}")
        print(f"  Monthly Expenses:      ${monthly_expenses:.2f}")
        print(f"  Current Savings:       ${current_savings:.2f} ({savings_rate:.1f}%)")
        
        print(f"\nRecommended Budget (50/30/20 Rule):")
        print(f"  Needs (50%):           ${recommended_needs:.2f}")
        print(f"  Wants (30%):           ${recommended_wants:.2f}")
        print(f"  Savings (20%):         ${recommended_savings:.2f}")
        
        print(f"\nPersonalized Recommendations:")
        
        recommendations = []
        
        # Savings recommendation
        if current_savings < recommended_savings:
            deficit = recommended_savings - current_savings
            recommendations.append(
                f"  1. Increase savings by ${deficit:.2f}/month to reach 20% savings goal"
            )
        else:
            recommendations.append(
                f"  1. Great job! You're saving ${current_savings:.2f}/month ({savings_rate:.1f}%)"
            )
        
        # Category-specific recommendations
        high_spending_categories = category_avg[category_avg > category_avg.median() * 1.5].sort_values(ascending=False)
        
        if len(high_spending_categories) > 0:
            for i, (cat, amt) in enumerate(high_spending_categories.head(3).items(), 2):
                reduce_by = amt * 0.15  # Suggest 15% reduction
                recommendations.append(
                    f"  {i}. Consider reducing {cat} spending by ${reduce_by:.2f}/month (15%)"
                )
        
        # Additional tips
        if savings_rate < 10:
            recommendations.append(
                f"  {len(recommendations)+1}. Track daily expenses to identify saving opportunities"
            )
        
        recommendations.append(
            f"  {len(recommendations)+1}. Review subscriptions and cancel unused services"
        )
        
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def calculate_financial_health_score(self):
        """Calculate overall financial health score (0-100)"""
        print("\n" + "="*70)
        print("FINANCIAL HEALTH SCORE")
        print("="*70)
        
        expenses = self.transactions[self.transactions['type'] == 'Expense'].copy()
        income = self.transactions[self.transactions['type'] == 'Income'].copy()
        
        # Ensure date is datetime
        expenses['date'] = pd.to_datetime(expenses['date'])
        income['date'] = pd.to_datetime(income['date'])
        
        monthly_income = income.groupby(income['date'].dt.to_period('M'))['amount'].sum().mean()
        monthly_expenses = expenses.groupby(expenses['date'].dt.to_period('M'))['amount'].sum().abs().mean()
        
        current_savings = monthly_income - monthly_expenses
        savings_rate = (current_savings / monthly_income) * 100 if monthly_income > 0 else 0
        
        # Calculate score components
        savings_score = min(savings_rate * 2, 40)  # Max 40 points
        expense_ratio = (monthly_expenses / monthly_income) * 100 if monthly_income > 0 else 100
        expense_score = max(0, (100 - expense_ratio) * 0.4)  # Max 40 points
        
        # Spending consistency (lower variance is better)
        monthly_exp = expenses.groupby(expenses['date'].dt.to_period('M'))['amount'].sum().abs()
        consistency_score = max(0, 20 - (monthly_exp.std() / monthly_exp.mean()) * 100)
        
        total_score = min(100, savings_score + expense_score + consistency_score)
        
        print(f"\nYour Financial Health Score: {total_score:.0f}/100\n")
        
        if total_score >= 80:
            status = "Excellent"
        elif total_score >= 60:
            status = "Good"
        elif total_score >= 40:
            status = "Fair"
        else:
            status = "Needs Improvement"
        
        print(f"  Status: {status}")
        print(f"\n  Breakdown:")
        print(f"    Savings Rate:         {savings_score:.0f}/40 points")
        print(f"    Expense Management:   {expense_score:.0f}/40 points")
        print(f"    Spending Consistency: {consistency_score:.0f}/20 points")
        
        return total_score
    
    def visualize_finances(self):
        """Create visualizations of financial data"""
        print("\nGenerating visualizations...")
        
        expenses = self.transactions[self.transactions['type'] == 'Expense'].copy()
        income = self.transactions[self.transactions['type'] == 'Income'].copy()
        
        # Ensure date is datetime
        expenses['date'] = pd.to_datetime(expenses['date'])
        income['date'] = pd.to_datetime(income['date'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Personal Finance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Monthly Income vs Expenses
        monthly_exp = expenses.groupby(expenses['date'].dt.to_period('M'))['amount'].sum().abs()
        monthly_inc = income.groupby(income['date'].dt.to_period('M'))['amount'].sum()
        
        ax1 = axes[0, 0]
        x = range(len(monthly_exp))
        ax1.plot(x, monthly_inc.values, marker='o', label='Income', linewidth=2, color='green')
        ax1.plot(x, monthly_exp.values, marker='o', label='Expenses', linewidth=2, color='red')
        ax1.fill_between(x, monthly_inc.values, monthly_exp.values, alpha=0.3, color='green')
        ax1.set_title('Monthly Income vs Expenses')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Amount ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spending by Category
        ax2 = axes[0, 1]
        category_spending = expenses.groupby('category')['amount'].sum().abs().sort_values(ascending=True)
        colors = plt.cm.Set3(range(len(category_spending)))
        category_spending.plot(kind='barh', ax=ax2, color=colors)
        ax2.set_title('Total Spending by Category')
        ax2.set_xlabel('Amount ($)')
        
        # 3. Category Distribution (Pie Chart)
        ax3 = axes[1, 0]
        top_categories = expenses.groupby('category')['amount'].sum().abs().nlargest(6)
        ax3.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Spending Distribution (Top Categories)')
        
        # 4. Daily Spending Trend
        ax4 = axes[1, 1]
        daily_spending = expenses.groupby('date')['amount'].sum().abs()
        daily_spending.plot(ax=ax4, color='purple', alpha=0.6)
        ax4.set_title('Daily Spending Trend')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Amount ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('finance_dashboard.png', dpi=300, bbox_inches='tight')
        print("Dashboard saved as 'finance_dashboard.png'")
        plt.show()
    
    def interactive_menu(self):
        """Interactive menu for user"""
        while True:
            print("\n" + "="*70)
            print("PERSONAL FINANCE ASSISTANT - MENU")
            print("="*70)
            print("\n1. View Spending Analysis")
            print("2. Predict Next Month Expenses")
            print("3. Predict Custom Month/Multiple Months")
            print("4. Detect Unusual Transactions")
            print("5. Get Budget Recommendations")
            print("6. Check Financial Health Score")
            print("7. Generate Visual Dashboard")
            print("8. Run Full Analysis")
            print("9. Exit")
            
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == '1':
                self.analyze_spending_patterns()
            elif choice == '2':
                next_month = (datetime.now().month % 12) + 1
                self.predict_expenses(next_month)
            elif choice == '3':
                self.predict_custom_scenario()
            elif choice == '4':
                self.detect_anomalies()
            elif choice == '5':
                self.generate_budget_recommendations()
            elif choice == '6':
                self.calculate_financial_health_score()
            elif choice == '7':
                self.visualize_finances()
            elif choice == '8':
                self.run_full_analysis()
            elif choice == '9':
                print("\nThank you for using Personal Finance Assistant!")
                break
            else:
                print("\nInvalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
    
    def run_full_analysis(self):
        """Run complete financial analysis"""
        print("\n" + "="*70)
        print("PERSONAL FINANCE ASSISTANT - FULL ANALYSIS")
        print("="*70)
        
        # Load data
        self.load_or_generate_data()
        
        # Run analyses
        self.analyze_spending_patterns()
        self.train_prediction_model()
        
        # Ask user for prediction
        print("\nWould you like to predict expenses? (yes/no)")
        response = input("> ").strip().lower()
        if response in ['yes', 'y']:
            self.predict_custom_scenario()
        
        self.detect_anomalies()
        self.generate_budget_recommendations()
        score = self.calculate_financial_health_score()
        
        # Visualize
        print("\nWould you like to generate visualizations? (yes/no)")
        response = input("> ").strip().lower()
        if response in ['yes', 'y']:
            try:
                self.visualize_finances()
            except Exception as e:
                print(f"Visualization error: {e}")
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70 + "\n")
        
        return score


def main():
    """Main function"""
    assistant = PersonalFinanceAssistant()
    
    # Load data first
    assistant.load_or_generate_data()
    
    # Train model
    assistant.train_prediction_model()
    
    # Start interactive menu
    assistant.interactive_menu()


if __name__ == "__main__":
    main()
