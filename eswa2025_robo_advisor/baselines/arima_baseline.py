import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(data):
    """
    数据预处理：对数变换和归一化
    """
    # 对数变换
    log_data = np.log(data)
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(log_data.reshape(-1, 1)).flatten()
    
    return normalized_data, scaler

def inverse_transform(data, scaler):
    """
    将归一化的数据转换回原始尺度
    """
    # 反归一化
    denormalized_data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    # 指数变换
    original_scale_data = np.exp(denormalized_data)
    return original_scale_data

def calculate_metrics(capital_series, initial_capital, trading_days, risk_free_rate=0.04):
    """
    计算交易性能指标
    """
    final_capital = capital_series[-1]
    total_profit = final_capital - initial_capital
    total_return = total_profit / initial_capital
    annual_return = (total_profit / initial_capital / trading_days) * 365 * 100
    daily_returns = np.diff(capital_series) / capital_series[:-1]
    
    peak = capital_series[0]
    mdd = 0
    for value in capital_series:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        mdd = max(mdd, dd)
    
    excess_returns = daily_returns - risk_free_rate/250
    sharpe = np.sqrt(250) * np.mean(excess_returns) / np.std(daily_returns)
    volatility = np.std(daily_returns) * np.sqrt(250)
    risk_return_ratio = volatility / (annual_return/100)
    
    return {
        'Total Profit': total_profit,
        'Total Return(%)': total_return * 100,
        'Annual Return(%)': annual_return,
        'MDD(%)': mdd * 100,
        'Sharpe Ratio': sharpe,
        'Risk/Return': risk_return_ratio,
        'Trading Days': trading_days,
        'Final Capital': final_capital
    }

def find_best_arima_params(train_data):
    """
    使用AIC准则选择最优的ARIMA模型参数
    """
    best_aic = float('inf')
    best_params = None
    
    # 修改参数范围，确保模型有最小复杂度
    p_range = range(0, 6)  # 包含0，允许更简单的模型
    d_range = range(0, 2)
    q_range = range(0, 6)  # 包含0，允许更简单的模型
    
    print("开始搜索最优ARIMA参数...")
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:  # 跳过过于简单的模型
                    continue
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        print(f"找到更优参数: ARIMA{best_params}, AIC: {best_aic:.2f}")
                        
                except:
                    continue
    
    print(f"\n最终选择的ARIMA参数: {best_params}, AIC: {best_aic:.2f}")
    return best_params

def generate_labels(prices, window_size=3):
    """
    使用3天窗口生成标签，增加对相同值的处理
    """
    results = []
    for i in range(window_size-1, len(prices)):
        window = prices[i-window_size+1:i+1]
        mid_index = window_size//2
        min_price = np.min(window)
        max_price = np.max(window)
        
        # 如果窗口内所有值相同，标记为持有
        if min_price == max_price:
            results.append(1)  # hold signal (0,1,0)
        else:
            if window[mid_index] == max_price:
                results.append(0)  # sell signal (1,0,0)
            elif window[mid_index] == min_price:
                results.append(2)  # buy signal (0,0,1)
            else:
                results.append(1)  # hold signal (0,1,0)
    return results

def arima_rolling_predict(file_path, initial_money, train_ratio):
    """
    使用滑动窗口的ARIMA预测，与BiGRU保持一致
    """
    # 读取数据
    print(f"正在读取文件: {file_path}")
    df = pd.read_csv(file_path)
    close_prices = df.iloc[:, 4].values
    print(f"数据长度: {len(close_prices)}")
    
    # 数据预处理
    normalized_prices, scaler = preprocess_data(close_prices)
    print("\n数据预处理:")
    print(f"原始数据范围: {min(close_prices):.2f} - {max(close_prices):.2f}")
    print(f"归一化后范围: {min(normalized_prices):.4f} - {max(normalized_prices):.4f}")
    
    # 划分初始训练集和测试集
    train_size = int(len(normalized_prices) * train_ratio)
    test_size = len(normalized_prices) - train_size
    period = 90  # 每90天为一个预测周期
    iteration = int((len(normalized_prices) - train_size)/period)  # 与BiGRU一致，使用向下取整
    print(f"训练集初始大小: {train_size}, 测试集大小: {test_size}")
    print(f"预测周期数: {iteration}")
    
    # 存储所有预测结果
    all_predictions = []
    all_test_labels = []
    all_pred_labels = []
    current = 0
    
    # 生成测试集的真实标签（使用原始价格）
    test_true_labels = generate_labels(close_prices[train_size:])
    
    for j in range(iteration):
        print(f"\n开始第 {j+1}/{iteration} 个预测周期")
        
        # 获取当前周期的训练数据
        train_data = normalized_prices[current:train_size+current]
        
        # 获取当前周期的测试数据（与BiGRU完全一致的逻辑）
        if j + 1 == iteration:  # 最后一个周期，预测所有剩余数据
            test_data = normalized_prices[train_size+current:]
            test_original = close_prices[train_size+current:]
            period_labels = test_true_labels[current:current+len(test_data)]
        else:  # 完整周期，预测90天
            test_data = normalized_prices[train_size+current:train_size+current+period]
            test_original = close_prices[train_size+current:train_size+current+period]
            period_labels = test_true_labels[current:current+period]
        
        # 使用AIC选择最优ARIMA参数
        best_params = find_best_arima_params(train_data)
        
        # ARIMA模型训练和预测
        print(f"使用最优参数 ARIMA{best_params} 训练模型...")
        model = ARIMA(train_data, order=best_params)
        results = model.fit()
        
        # 预测当前周期的收盘价
        predictions_normalized = results.forecast(steps=len(test_data))
        predictions = inverse_transform(predictions_normalized, scaler)
        
        # 生成当前周期的交易信号
        pred_labels = generate_labels(predictions)
        
        # 确保预测标签和实际标签长度匹配
        min_len = min(len(period_labels), len(pred_labels))
        period_labels = period_labels[:min_len]
        pred_labels = pred_labels[:min_len]
        predictions = predictions[:min_len]
        test_original = test_original[:min_len]
        
        # 存储结果
        all_predictions.extend(predictions)
        all_test_labels.extend(period_labels)
        all_pred_labels.extend(pred_labels)
        
        # 更新窗口位置
        current += min_len  # 更新为实际处理的长度
        
        print(f"第 {j+1} 个周期预测完成")
        print(f"当前周期标签数量 - 预测: {len(pred_labels)}, 实际: {len(period_labels)}")
    
    # 确保最终的标签长度匹配
    print(f"\n标签长度检查:")
    print(f"预测标签数量: {len(all_pred_labels)}")
    print(f"实际标签数量: {len(all_test_labels)}")
    
    # 计算整体预测指标
    accuracy = accuracy_score(all_test_labels, all_pred_labels)
    f1 = f1_score(all_test_labels, all_pred_labels, average='weighted')
    recall = recall_score(all_test_labels, all_pred_labels, average='weighted')
    precision = precision_score(all_test_labels, all_pred_labels, average='weighted')
    rmse = np.sqrt(mean_squared_error(all_test_labels, all_pred_labels))
    mae = mean_absolute_error(all_test_labels, all_pred_labels)
    
    # 计算R2分数
    r2 = r2_score(all_test_labels, all_pred_labels)
    
    print("\n整体预测性能:")
    print(f"预测准确率: {accuracy*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # 添加标签统计
    print("\n标签统计:")
    print("实际标签:")
    print(f"卖出信号数量: {all_test_labels.count(0)}")
    print(f"持有信号数量: {all_test_labels.count(1)}")
    print(f"买入信号数量: {all_test_labels.count(2)}")
    print("\n预测标签:")
    print(f"卖出信号数量: {all_pred_labels.count(0)}")
    print(f"持有信号数量: {all_pred_labels.count(1)}")
    print(f"买入信号数量: {all_pred_labels.count(2)}")
    
    return close_prices, all_pred_labels, train_size, {
        'Accuracy': accuracy,
        'F1_Score': f1,
        'Recall': recall,
        'Precision': precision,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'ARIMA_params': str(best_params),
        'AIC': results.aic
    }

def simulate_trading(close_prices, signals, train_size, initial_money):
    """
    交易模拟
    """
    print(f"开始交易模拟，初始资金: {initial_money}")
    money = initial_money
    stock_value = 0
    volume = 0
    capital_series = [initial_money]
    i = 0
    total_trades = 0
    
    # 记录每笔交易的详细信息
    trades_log = []
    
    while i <= (len(signals)-2): 
        if i % 100 == 0:
            print(f"模拟进度: {i}/{len(signals)}")
            
        current_price = close_prices[train_size+i]
        
        if signals[i] != 2:  # 不是买入信号
            stock_value = volume * current_price
            capital = money + stock_value        
            i = i + 1
            capital_series.append(capital)
        else:  # 买入信号
            volume = int(money/current_price)
            stock_value = current_price * volume
            buy_brokerage = max(stock_value * 0.001, 5)
            money = money - stock_value - buy_brokerage
            capital = money + stock_value
            capital_series.append(capital)
            
            # 记录买入交易
            trades_log.append({
                'type': 'buy',
                'price': current_price,
                'volume': volume,
                'value': stock_value,
                'fee': buy_brokerage,
                'capital': capital
            })
            
            j = i + 1
            active = True        
            while active:
                if j == (len(signals)-1):
                    i = j
                    active = False
                    break
                    
                current_price = close_prices[train_size+j]
                if signals[j] != 0:  # 不是卖出信号
                    stock_value = volume * current_price
                    capital = money + stock_value        
                    capital_series.append(capital)
                    j = j + 1
                    if j == (len(signals)-1):
                        i = j
                        active = False
                else:  # 卖出信号
                    potential_profit = (current_price - close_prices[train_size+i])*volume
                    potential_sell_brokerage = current_price * volume * 0.001
                    potential_stamp_tax = current_price * volume * 0.0005
                    potential_sell_cost = potential_sell_brokerage + potential_stamp_tax

                    if potential_profit > (buy_brokerage + potential_sell_cost):
                        total_trades = total_trades + 1 
                        sell_brokerage = max(potential_sell_brokerage, 5)
                        stamp_tax = potential_stamp_tax
                        sell_cost = sell_brokerage + stamp_tax
                        stock_value = current_price * volume
                        money = money + stock_value - sell_cost
                        capital = money
                        capital_series.append(capital)
                        
                        # 记录卖出交易
                        trades_log.append({
                            'type': 'sell',
                            'price': current_price,
                            'volume': volume,
                            'value': stock_value,
                            'fee': sell_cost,
                            'profit': potential_profit - (buy_brokerage + sell_cost),
                            'capital': capital
                        })
                    else:
                        stock_value = volume * current_price
                        capital = money + stock_value        
                        capital_series.append(capital)

                    i = j + 1
                    stock_value = 0
                    volume = 0
                    money = capital
                    active = False
    
    print(f"交易模拟完成，总交易次数: {total_trades}")
    print(f"最终资金: {capital_series[-1]}")
    
    # 打印交易统计信息
    if trades_log:
        total_profit = sum(trade['profit'] for trade in trades_log if 'profit' in trade)
        total_fees = sum(trade['fee'] for trade in trades_log)
        print(f"总收益: {total_profit:.2f}")
        print(f"总手续费: {total_fees:.2f}")
    
    return capital_series, total_trades

def _jobs(data_dir):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from eswa2025_robo_advisor.config import ASSETS

    jobs = []
    data_dir = Path(data_dir)
    for name, cfg in ASSETS.items():
        found = None
        for stem in cfg["file_stems"]:
            candidate = data_dir / ("%s_index.csv" % stem)
            if candidate.exists():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError("missing index csv for %s in %s" % (name, data_dir))
        jobs.append((str(found), cfg["initial_capital"], cfg["train_ratio"], name))
    return jobs


def run(data_dir, output_csv="arima_trading_results.csv"):
    results = {}
    for file_path, initial_money, train_ratio, asset_name in _jobs(data_dir):
        print("\n" + "=" * 50)
        print("处理文件: %s" % file_path)
        close_prices, signals, train_size, prediction_metrics = arima_rolling_predict(
            file_path, initial_money, train_ratio
        )
        capital_series, total_trades = simulate_trading(close_prices, signals, train_size, initial_money)
        trading_days = len(capital_series)
        trading_metrics = calculate_metrics(capital_series, initial_money, trading_days)
        results[asset_name] = {**trading_metrics, **prediction_metrics, "Trades": total_trades}
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_csv, float_format="%.4f")
    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARIMA baseline in Table 3 and Table 4")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="arima_trading_results.csv")
    args = parser.parse_args()
    print(run(args.data_dir, args.output))