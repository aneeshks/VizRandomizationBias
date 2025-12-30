import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.signal import savgol_filter

plt.style.use('seaborn-v0_8')

from datetime import datetime, timedelta

def create_customer_segments(n_customers):
    """Create customer IDs and assign segments with equal proportions (33% each)"""
    customer_ids = [f'CUST_{str(i).zfill(7)}' for i in range(n_customers)]
    
    segments = ['Low_CTR'] * (n_customers//3) + ['Medium_CTR'] * (n_customers//3) + ['High_CTR'] * (n_customers//3)
    while len(segments) < n_customers:
        segments.append('Medium_CTR')
    
    np.random.shuffle(segments)
    return dict(zip(customer_ids, segments))

def generate_data(customer_segments, 
                 start_date, 
                 end_date, 
                 ctr_probs,
                 treatment_effects,
                 treatment_mix={'High_CTR': 0.5, 'Medium_CTR': 0.5, 'Low_CTR': 0.5},
                 views_range=(1, 1000),
                 n_products=1000,
                 click_delay_range=(1, 604800)):
    """
    Generate data with specific composition for treatment and control groups
    
    Parameters:
    -----------
    customer_segments : dict
        Mapping of customer IDs to their segments
    start_date : datetime
        Start date for generating view times
    end_date : datetime
        End date for generating view times
    ctr_probs : dict
        Base CTR probabilities for each segment
    treatment_effects : dict
        Treatment effects for each segment
    treatment_mix : dict
        Proportion of treatment assignment for each segment
    views_range : tuple
        Range of number of views per customer (min, max)
    n_products : int
        Number of unique products
    click_delay_range : tuple
        Range of delay between view and click in seconds (min, max)
    
    Returns:
    --------
    pd.DataFrame
        Generated click data
    """
    total_seconds = int((end_date - start_date).total_seconds())
    product_ids = [f'P{str(i).zfill(4)}' for i in range(1, n_products + 1)]
    
    # Get customers by segment
    segment_customers = {
        segment: [cust for cust, seg in customer_segments.items() if seg == segment]
        for segment in ['Low_CTR', 'Medium_CTR', 'High_CTR']
    }
    
    # Print available customers per segment
    print("\nAvailable customers per segment:")
    for segment, customers in segment_customers.items():
        print(f"{segment}: {len(customers)}")
    
    # Calculate number of customers needed for treatment
    min_segment_size = min(len(customers) for customers in segment_customers.values())
    print(f"Minimum segment size: {min_segment_size}")
    total_treatment = min_segment_size
    
    # Calculate and adjust treatment counts
    treatment_counts = {
        segment: min(int(total_treatment * prop), len(segment_customers[segment]))
        for segment, prop in treatment_mix.items()
    }
    
    # Assign treatments
    treatment_assignments = {}
    for segment, count in treatment_counts.items():
        available_customers = segment_customers[segment]
        t_customers = np.random.choice(available_customers, size=count, replace=False)
        c_customers = list(set(available_customers) - set(t_customers))
        
        for cust in t_customers:
            treatment_assignments[cust] = 'T'
        for cust in c_customers:
            treatment_assignments[cust] = 'C'
    
    # Generate data
    data = []
    for cust_id, segment in customer_segments.items():
        if cust_id not in treatment_assignments:
            continue
            
        group = treatment_assignments[cust_id]
        n_views = np.random.randint(views_range[0], views_range[1])
        products = np.random.choice(product_ids, n_views, replace=True)
        
        view_times = sorted([start_date + timedelta(seconds=np.random.randint(0, total_seconds)) 
                           for _ in range(n_views)])
        
        trigger_index = np.random.randint(0, n_views)
        trigger_time = view_times[trigger_index]
        
        for idx, (prod, view_time) in enumerate(zip(products, view_times)):
            base_click_prob = ctr_probs[segment]
            is_post_trigger = view_time >= trigger_time
            
            click_prob = base_click_prob + treatment_effects[segment] if is_post_trigger and group == 'T' else base_click_prob
            
            if np.random.random() < click_prob:
                click_delay = np.random.randint(click_delay_range[0], click_delay_range[1])
                click_time = min(view_time + timedelta(seconds=click_delay), end_date)
            else:
                click_time = None
            
            data.append({
                'customer_id': cust_id,
                'product_id': prod,
                'view_time': view_time,
                'click_time': click_time,
                'customer_segment': segment,
                'group': group,
                'trigger_time': trigger_time
            })
    
    return pd.DataFrame(data)



def print_basic_statistics(df):
    """Print basic statistics about the dataset."""
    unique_customers = df['customer_id'].nunique()
    print("="*80)
    print("BASIC STATISTICS")
    print("="*80)
    print(f"Number of unique customers: {unique_customers}")
    print("\nData Shape:", df.shape)
    
    print("\nCustomer Segments Distribution:")
    segment_dist = df.groupby('customer_id')['customer_segment'].first().value_counts(normalize=True)
    print(segment_dist.round(3))
    
    print("\nTreatment Assignment by Segment:")
    treatment_dist = df.groupby(['customer_segment', 'group'])['customer_id'].nunique()
    print(treatment_dist)
    
    print("\nTreatment Ratios by Segment:")
    treatment_ratios = (df.groupby(['customer_segment', 'group'])['customer_id']
                       .nunique()
                       .unstack(fill_value=0))
    treatment_ratios['total'] = treatment_ratios.sum(axis=1)
    treatment_ratios['T_ratio'] = (treatment_ratios['T'] / treatment_ratios['total']).round(3)
    print(treatment_ratios)

def calculate_ctr_metrics(df):
    """Calculate and print CTR metrics."""
    df['is_post_trigger'] = df['view_time'] >= df['trigger_time']
    
    print("\n")
    print("="*80)
    print("CTR ANALYSIS")
    print("="*80)
    
    overall_ctr = len(df[df['click_time'].notna()]) / len(df) * 100
    print("\nOverall CTR:")
    print(f"{overall_ctr:.3f}%")
    
    print("\nCTR by Segment:")
    segment_ctr = df.groupby('customer_segment').agg(
        total_views=('product_id', 'count'),
        total_clicks=('click_time', 'count')
    )
    segment_ctr['CTR'] = (segment_ctr['total_clicks'] / segment_ctr['total_views'] * 100).round(3)
    print(segment_ctr)
    
    print("\nCTR by Treatment Group:")
    group_ctr = df.groupby('group').agg(
        total_views=('product_id', 'count'),
        total_clicks=('click_time', 'count')
    )
    group_ctr['CTR'] = (group_ctr['total_clicks'] / group_ctr['total_views'] * 100).round(3)
    print(group_ctr)
    
    return df

def calculate_detailed_ctr(df):
    """Calculate detailed CTR statistics."""
    print("\n")
    print("="*80)
    print("DETAILED CTR ANALYSIS BY SEGMENT, GROUP, AND TRIGGER PERIOD")
    print("="*80)
    
    ctr_stats = df.groupby(['customer_segment', 'group', 'is_post_trigger']).agg(
        total_views=('product_id', 'count'),
        total_clicks=('click_time', 'count')
    ).reset_index()
    ctr_stats['CTR'] = (ctr_stats['total_clicks'] / ctr_stats['total_views'] * 100).round(3)
    print("\nDetailed CTR Statistics:")
    print(ctr_stats)
    
    return ctr_stats

def calculate_treatment_effect(data):
    """Calculate treatment effects."""
    pivot = data.pivot(
        index=['customer_segment', 'is_post_trigger'],
        columns='group',
        values='CTR'
    ).reset_index()
    
    pivot['T-C_diff'] = pivot['T'] - pivot['C']
    
    sizes = data.pivot_table(
        index=['customer_segment', 'is_post_trigger'],
        columns='group',
        values='total_views',
        aggfunc='sum'
    ).reset_index()
    
    pivot['T_views'] = sizes['T']
    pivot['C_views'] = sizes['C']
    
    pivot['T_se'] = np.sqrt((pivot['T']*(100-pivot['T']))/(pivot['T_views']))
    pivot['C_se'] = np.sqrt((pivot['C']*(100-pivot['C']))/(pivot['C_views']))
    pivot['diff_se'] = np.sqrt(pivot['T_se']**2 + pivot['C_se']**2)
    
    return pivot

def print_treatment_effects(treatment_effects):
    """Print treatment effects analysis."""
    print("\n")
    print("="*80)
    print("TREATMENT EFFECTS")
    print("="*80)
    
    for segment in treatment_effects['customer_segment'].unique():
        print(f"\n{segment}:")
        segment_data = treatment_effects[treatment_effects['customer_segment'] == segment]
        
        pre_effect = segment_data[segment_data['is_post_trigger'] == False]['T-C_diff'].iloc[0]
        pre_se = segment_data[segment_data['is_post_trigger'] == False]['diff_se'].iloc[0]
        
        post_effect = segment_data[segment_data['is_post_trigger'] == True]['T-C_diff'].iloc[0]
        post_se = segment_data[segment_data['is_post_trigger'] == True]['diff_se'].iloc[0]
        
        treatment_effect = post_effect - pre_effect
        treatment_se = np.sqrt(pre_se**2 + post_se**2)
        
        print(f"Pre-trigger difference (T-C): {pre_effect:.3f}% ± {2*pre_se:.3f}%")
        print(f"Post-trigger difference (T-C): {post_effect:.3f}% ± {2*post_se:.3f}%")
        print(f"Treatment Effect: {treatment_effect:.3f}% ± {2*treatment_se:.3f}%")

def create_summary_table(treatment_effects):
    """Create and print summary table."""
    summary_table = pd.DataFrame({
        'Segment': treatment_effects['customer_segment'].unique(),
        'Pre_Trigger_Diff': treatment_effects[~treatment_effects['is_post_trigger']]['T-C_diff'].values,
        'Post_Trigger_Diff': treatment_effects[treatment_effects['is_post_trigger']]['T-C_diff'].values
    })
    summary_table['Treatment_Effect'] = summary_table['Post_Trigger_Diff'] - summary_table['Pre_Trigger_Diff']
    
    print("\n")
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_table.round(3))
    
    return summary_table

def analyze_post_trigger_difference(df):
    """Analyze post-trigger differences between treatment and control."""
    print("\n")
    print("="*80)
    print("OVERALL POST-TRIGGER DIFFERENCE (T vs C)")
    print("="*80)
    
    post_trigger_data = df[df['is_post_trigger']]
    overall_post_stats = post_trigger_data.groupby('group').agg(
        total_views=('product_id', 'count'),
        total_clicks=('click_time', 'count')
    )
    overall_post_stats['CTR'] = (overall_post_stats['total_clicks'] / overall_post_stats['total_views'] * 100)
    
    T_ctr = overall_post_stats.loc['T', 'CTR']
    C_ctr = overall_post_stats.loc['C', 'CTR']
    T_views = overall_post_stats.loc['T', 'total_views']
    C_views = overall_post_stats.loc['C', 'total_views']
    
    T_se = np.sqrt((T_ctr*(100-T_ctr))/T_views)
    C_se = np.sqrt((C_ctr*(100-C_ctr))/C_views)
    diff_se = np.sqrt(T_se**2 + C_se**2)
    
    overall_diff = T_ctr - C_ctr
    
    print("\nPost-Trigger Statistics:")
    print(f"Treatment CTR: {T_ctr:.3f}% (n={T_views:,})")
    print(f"Control CTR: {C_ctr:.3f}% (n={C_views:,})")
    print(f"\nAbsolute Difference (T-C): {overall_diff:.3f}% ± {2*diff_se:.3f}%")
    print(f"Relative Difference: {((T_ctr/C_ctr - 1) * 100):.2f}%")
    
    z_score = overall_diff / diff_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    print(f"P-value: {p_value:.4f}")




def prepare_time_based_analysis(df, n_minutes=720):  # 720 minutes = 12 hours
    """
    Prepare time-based CTR analysis data
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame with view_time and trigger_time
    n_minutes : int
        Number of minutes to analyze before and after trigger
    
    Returns:
    --------
    tuple : (pivot_df_window, se)
        Filtered pivot table and standard error calculations
    """
    # Convert time to minutes and round
    df['diff_time'] = (df['view_time'] - df['trigger_time']).dt.total_seconds()
    df['diff_units'] = (df['diff_time']/60).round(0)
    
    # Calculate CTR for each time unit and group
    summary_df = (df
        .groupby(['diff_units', 'group'])
        .agg(
            total_views=('customer_id', 'count'),
            total_clicks=('click_time', 'count')
        )
        .reset_index()
    )
    
    # Split into positive and negative time periods
    pos_df = summary_df[summary_df['diff_units'] >= 0].sort_values('diff_units')
    neg_df = summary_df[summary_df['diff_units'] < 0].sort_values('diff_units', ascending=False)
    
    cumulative_stats = []
    for group in ['C', 'T']:
        # Forward cumulative (0 to positive)
        pos_data = pos_df[pos_df['group'] == group].copy()
        pos_data['cum_views'] = pos_data['total_views'].cumsum()
        pos_data['cum_clicks'] = pos_data['total_clicks'].cumsum()
        pos_data['cum_CTR'] = np.where(
            pos_data['cum_views'] > 0,
            (pos_data['cum_clicks']*100 / pos_data['cum_views']),
            0
        )
        
        # Backward cumulative (0 to negative)
        neg_data = neg_df[neg_df['group'] == group].copy()
        neg_data['cum_views'] = neg_data['total_views'].cumsum()
        neg_data['cum_clicks'] = neg_data['total_clicks'].cumsum()
        neg_data['cum_CTR'] = np.where(
            neg_data['cum_views'] > 0,
            (neg_data['cum_clicks']*100 / neg_data['cum_views']),
            0
        )
        
        group_data = pd.concat([neg_data, pos_data]).sort_values('diff_units')
        cumulative_stats.append(group_data)
    
    cum_summary_df = pd.concat(cumulative_stats)
    
    # Create pivot table
    pivot_df = cum_summary_df.pivot(index='diff_units', 
                                   columns='group', 
                                   values=['cum_CTR'])
    
    pivot_df['CTR_diff'] = pivot_df[('cum_CTR', 'T')] - pivot_df[('cum_CTR', 'C')]
    
    # Filter data for the specified time window
    pivot_df_window = pivot_df[
        (pivot_df.index >= -n_minutes) & 
        (pivot_df.index <= n_minutes)
    ].copy()
    
    # Calculate standard errors
    t_data = cum_summary_df[cum_summary_df['group'] == 'T'].set_index('diff_units')
    c_data = cum_summary_df[cum_summary_df['group'] == 'C'].set_index('diff_units')
    
    t_p = t_data['cum_clicks'] / t_data['cum_views']
    c_p = c_data['cum_clicks'] / c_data['cum_views']
    t_se = np.sqrt((t_p * (1-t_p)) / t_data['cum_views'])
    c_se = np.sqrt((c_p * (1-c_p)) / c_data['cum_views'])
    se = np.sqrt(t_se**2 + c_se**2) * 100  # Convert to percentage
    
    return pivot_df_window, se


def plot_time_based_ctr(pivot_df_window, se, n_minutes=720):
    """
    Plot time-based CTR difference with confidence intervals with symmetric bounds around 0
    """
    # Set style parameters
    plt.rcParams.update({
        'font.size': 6,          # Base font size
        'axes.labelsize': 6,     # Axis labels
        'axes.titlesize': 6,     # Title
        'xtick.labelsize': 5,    # X-axis tick labels
        'ytick.labelsize': 5,    # Y-axis tick labels
        'legend.fontsize': 5,    # Legend
    })
    
    # Create figure
    plt.figure(figsize=(5, 3), dpi=300)
    
    # Calculate absolute max and min including confidence intervals
    lower_ci = pivot_df_window['CTR_diff'] - 2*se[pivot_df_window.index]
    upper_ci = pivot_df_window['CTR_diff'] + 2*se[pivot_df_window.index]
    
    absolute_min = lower_ci.min()
    absolute_max = upper_ci.max()
    
    # print(f"Absolute MIN (including CI): {absolute_min:.6f}")
    # print(f"Absolute MAX (including CI): {absolute_max:.6f}")
    
    # Find the maximum absolute value to ensure symmetry around 0
    max_abs_value = max(abs(absolute_min), abs(absolute_max))
    
    # Add small padding (5%)
    padding = max_abs_value * 0.05
    
    # Set symmetric bounds
    y_bound = max_abs_value + padding
    y_min_bound = -y_bound
    y_max_bound = y_bound
    
    # print(f"\nSymmetric bounds around 0:")
    # print(f"Lower bound: {y_min_bound:.6f}")
    # print(f"Upper bound: {y_max_bound:.6f}")
    
    # Plot CTR difference
    hours = n_minutes/60
    plt.plot(pivot_df_window.index/60, pivot_df_window['CTR_diff'], 
             color='#2878B5', linewidth=1.2, label='Mean CTR Difference')
    
    # Add confidence interval
    plt.fill_between(pivot_df_window.index/60, 
                     lower_ci,
                     upper_ci,
                     color='#2878B5', alpha=0.2, label='95% CI')
    
    # Explicitly set symmetric y-axis limits
    plt.ylim(y_min_bound, y_max_bound)
    
    # Print actual y-limits set
    current_ylim = plt.gca().get_ylim()
    print(f"\nActual y-limits set in plot:")
    print(f"Lower y-limit: {current_ylim[0]:.6f}")
    print(f"Upper y-limit: {current_ylim[1]:.6f}")
    
    # Add reference lines
    plt.axvline(x=0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axhline(y=0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Set x-axis ticks and limits
    plt.xlim(-hours, hours)
    xticks = np.arange(-hours, hours + 1, 2)
    plt.xticks(xticks, [f'{x:+}' for x in xticks])
    
    # Labels and formatting
    plt.xlabel('Time Relative to Trigger (Hours)')
    plt.ylabel('CTR Difference (%)')
    plt.title('Impact of Treatment on Click-Through Rate Over Time', pad=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return plt.gcf()


