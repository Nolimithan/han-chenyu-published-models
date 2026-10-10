import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(data):
    """
    数据预处理：对数变换和归一化
    """
    # 处理负值
    min_value = np.min(data)
    if min_value <= 0:
      shift = abs(min_value) + 1  # 确保所有值都为正
      data = data + shift
    print(f"数据平移: {shift:.2f}")
        
        # 对数变换
    log_data = np.log(data)
        
    # 计算收益率
    returns = np.diff(log_data)
    returns = np.insert(returns, 0, 0)  # 在开始处插入0，保持长度一致
    
    # 标准化收益率
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_returns = scaler.fit_transform(returns.reshape(-1, 1)).flatten()
    
    return normalized_returns, scaler, min_value <= 0, shift if min_value <= 0 else 0

def inverse_transform(data, scaler, has_shift=False, shift_value=0):
    """
    将归一化的数据转换回原始尺度
    """
    try:
        # 反归一化
        denormalized_returns = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        
        # 使用更合理的收益率限制
        std_returns = np.std(denormalized_returns)
        max_return = 3 * std_returns  # 使用3倍标准差作为限制
        denormalized_returns = np.clip(denormalized_returns, -max_return, max_return)
        
        # 计算累积收益率
        cumulative_returns = np.cumsum(denormalized_returns)
        
        # 使用指数函数将收益率转换为价格
        prices = np.exp(cumulative_returns)
        
        # 将价格缩放到合理范围
        price_scale = np.median(prices)
        if price_scale > 0:
            prices = prices / price_scale
            
        # 如果之前进行了平移，需要减去平移量
        if has_shift:
            prices = prices - shift_value
            
        return prices
        
    except Exception as e:
        print(f"转换过程出错: {str(e)}")
        return np.zeros_like(data)

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

def find_best_markov_params(train_data):
    """
    使用AIC准则选择最优的马尔科夫转换模型参数
    """
    best_aic = float('inf')
    best_params = None
    best_model = None
    
    # 设置参数搜索范围
    k_regimes_range = [2]  # 限制为2个状态
    order_range = range(1, 3)  # 限制AR阶数为1-2
    
    print("开始搜索最优马尔科夫转换模型参数...")
    
    # 添加数据检查
    if np.any(np.isnan(train_data)) or np.any(np.isinf(train_data)):
        print("警告：数据中包含NaN或Inf，将被替换")
        train_data = np.nan_to_num(train_data, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 标准化数据以提高数值稳定性
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    if train_std > 0:
        train_data_scaled = (train_data - train_mean) / train_std
    else:
        train_data_scaled = train_data - train_mean
    
    for k_regimes in k_regimes_range:
        for order in order_range:
            try:
                # 添加更多的初始化选项
                model = MarkovRegression(
                    endog=train_data_scaled,
                    k_regimes=k_regimes,
                    order=order,
                    switching_variance=True,
                    switching_trend=True
                )
                
                # 使用不同的优化方法
                for method in ['powell', 'bfgs', 'nm']:
                    try:
                        results = model.fit(
                            method=method,
                            maxiter=1000,
                            disp=0
                        )
                        
                        if results is not None and hasattr(results, 'aic'):
                            aic = results.aic
                            if aic < best_aic and not np.isnan(aic) and not np.isinf(aic):
                                best_aic = aic
                                best_params = (k_regimes, order)
                                best_model = results
                                print(f"找到更优参数: Regimes={k_regimes}, Order={order}, AIC: {aic:.2f}, Method={method}")
                                break  # 如果找到有效结果就跳出优化方法循环
                    except:
                        continue
            except:
                continue
    
    if best_params is None:
        # 如果所有参数组合都失败，使用最简单的模型
        try:
            model = MarkovRegression(
                endog=train_data_scaled,
                k_regimes=2,
                order=1,
                switching_variance=False,
                switching_trend=False
            )
            results = model.fit(disp=0)
            best_params = (2, 1)
            best_model = results
            print("使用默认参数的简单模型")
        except:
            raise Exception("未能找到有效的模型参数")
    
    print(f"\n最终选择的参数: Regimes={best_params[0]}, Order={best_params[1]}, AIC: {best_aic:.2f}")
    return best_params, best_model

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
    return np.array(results)  # 返回numpy数组而不是列表

def estimate_transition_matrix_mle(model_results, best_params):
    """
    使用最大似然估计计算状态转移概率矩阵
    """
    try:
        # 获取平滑概率
        if hasattr(model_results, 'smoothed_marginal_probabilities'):
            probs = model_results.smoothed_marginal_probabilities
        else:
            probs = model_results.filtered_marginal_probabilities
            
        if isinstance(probs, pd.DataFrame):
            probs = probs.values
            
        n_regimes = best_params[0]
        T = len(probs)
        
        # 初始化转移计数矩阵
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        # 计算每个时间点t的状态概率和t+1时间点的状态概率
        for t in range(T-1):
            # 获取t时刻和t+1时刻的状态概率
            prob_t = probs[t]
            prob_t1 = probs[t+1]
            
            # 计算联合概率矩阵
            joint_prob = np.outer(prob_t, prob_t1)
            
            # 累加转移计数
            transition_counts += joint_prob
        
        # 计算转移概率
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        # 避免除以0
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_counts / row_sums
        
        print("使用MLE估计的转移概率矩阵")
        return transition_matrix
        
    except Exception as e:
        print(f"MLE估计转移概率失败: {str(e)}")
        return None

def markov_trading(file_path, initial_money, train_ratio):
    try:
        # 读取数据
        print(f"正在读取文件: {file_path}")
        df = pd.read_csv(file_path)
        close_prices = df.iloc[:, 4].values
        print(f"数据长度: {len(close_prices)}")
        
        # 数据预处理
        normalized_returns, scaler, has_shift, shift_value = preprocess_data(close_prices)
        print("\n数据预处理:")
        print(f"原始数据范围: {min(close_prices):.2f} - {max(close_prices):.2f}")
        print(f"归一化后范围: {min(normalized_returns):.4f} - {max(normalized_returns):.4f}")
        
        # 划分初始训练集和测试集
        train_size = int(len(normalized_returns) * train_ratio)
        test_size = len(normalized_returns) - train_size
        period = 90  # 每90天为一个预测周期
        iteration = int((len(normalized_returns) - train_size)/period)  # 与BiGRU一致，使用向下取整
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
            train_data = normalized_returns[current:train_size+current]
            
            # 获取当前周期的测试数据（与BiGRU完全一致的逻辑）
            if j + 1 == iteration:  # 最后一个周期，预测所有剩余数据
                test_data = normalized_returns[train_size+current:]
                test_original = close_prices[train_size+current:]
                period_labels = test_true_labels[current:current+len(test_data)]
            else:  # 完整周期，预测90天
                test_data = normalized_returns[train_size+current:train_size+current+period]
                test_original = close_prices[train_size+current:train_size+current+period]
                period_labels = test_true_labels[current:current+period]
            
            # 使用AIC选择最优参数
            best_params, model_results = find_best_markov_params(train_data)
            
            # 获取最后一个状态的概率
            if hasattr(model_results, 'smoothed_marginal_probabilities'):
                probs = model_results.smoothed_marginal_probabilities
                if isinstance(probs, pd.DataFrame):
                    probs = probs.values
                current_regime = np.argmax(probs[-1])
            else:
                probs = model_results.filtered_marginal_probabilities
                if isinstance(probs, pd.DataFrame):
                    probs = probs.values
                current_regime = np.argmax(probs[-1])
            
            # 获取模型参数
            params_dict = {}
            n_params_per_regime = len(model_results.params) // best_params[0]
            
            # 遍历每个状态
            for regime in range(best_params[0]):
                regime_params = {}
                # 常数项
                regime_params['const'] = model_results.params[regime * n_params_per_regime]
                # AR项
                for k in range(best_params[1]):
                    regime_params[f'L{k+1}'] = model_results.params[regime * n_params_per_regime + k + 1]
                params_dict[regime] = regime_params
            
            # 获取转移概率矩阵
            transition_matrix = estimate_transition_matrix_mle(model_results, best_params)
            if transition_matrix is None:
                try:
                    transition_matrix = model_results.transitional_probabilities
                    if transition_matrix is None:
                        filtered_states = np.argmax(probs, axis=1)
                        transition_matrix = np.zeros((best_params[0], best_params[0]))
                        for i in range(len(filtered_states)-1):
                            current_state = filtered_states[i]
                            next_state = filtered_states[i+1]
                            transition_matrix[current_state, next_state] += 1
                        row_sums = transition_matrix.sum(axis=1)
                        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
                except AttributeError:
                    n_regimes = best_params[0]
                    transition_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
            
            # 创建预测窗口
            window = list(train_data[-best_params[1]:])
            predictions_normalized = []
            
            # 对当前周期进行预测
            for i in range(len(test_data)):
                try:
                    # 使用当前状态的参数进行预测
                    regime_params = params_dict[current_regime]
                    
                    # 计算预测值
                    prediction = regime_params.get('const', 0.0)
                    for k in range(min(len(window), best_params[1])):
                        param_name = f'L{k+1}'
                        if param_name in regime_params:
                            prediction += regime_params[param_name] * window[-(k+1)]
                    
                    # 使用sigmoid函数将预测值映射到(0,1)区间
                    prediction = 1 / (1 + np.exp(-prediction))
                    predictions_normalized.append(prediction)
                    
                    # 更新窗口
                    window.append(test_data[i])
                    window = window[1:]
                    
                    # 更新状态
                    if i < len(test_data) - 1:
                        trans_probs = transition_matrix[current_regime]
                        trans_probs = trans_probs / np.sum(trans_probs)
                        current_regime = np.random.choice(best_params[0], p=trans_probs)
                        
                except Exception as e:
                    print(f"预测第{i}个样本时出错: {str(e)}")
                    if len(predictions_normalized) > 0:
                        predictions_normalized.append(predictions_normalized[-1])
                    else:
                        predictions_normalized.append(test_data[i])
            
            # 将预测结果转换回原始尺度
            predictions = inverse_transform(np.array(predictions_normalized), scaler, has_shift, shift_value)
            
            # 检查预测结果的有效性
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print("警告：预测结果包含无效值，使用简单移动平均替代")
                window_size = 5
                predictions = pd.Series(test_original).rolling(window=window_size, center=True).mean().fillna(method='ffill').fillna(method='bfill').values
            
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
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_test_labels, all_pred_labels)
        
        # 计算每个类别的精确度和召回率
        class_precision = {}
        class_recall = {}
        for i in range(3):
            if cm[i].sum() != 0:
                class_precision[i] = cm[i,i] / cm[:,i].sum()
            else:
                class_precision[i] = 0
            if cm[:,i].sum() != 0:
                class_recall[i] = cm[i,i] / cm[i,:].sum()
            else:
                class_recall[i] = 0
        
        # 计算预测标签的统计特征
        pred_mean = np.mean(all_pred_labels)
        pred_std = np.std(all_pred_labels)
        pred_skew = np.mean(((all_pred_labels - pred_mean) / pred_std) ** 3) if pred_std != 0 else 0
        pred_kurtosis = np.mean(((all_pred_labels - pred_mean) / pred_std) ** 4) - 3 if pred_std != 0 else 0
        
        # 计算预测准确率的置信区间
        from scipy import stats
        confidence = 0.95
        n = len(all_test_labels)
        interval = stats.norm.interval(confidence, accuracy, np.sqrt((accuracy * (1 - accuracy)) / n))
        
        # 计算额外的统计指标
        rmse = np.sqrt(mean_squared_error(all_test_labels, all_pred_labels))
        mae = mean_absolute_error(all_test_labels, all_pred_labels)
        from sklearn.metrics import r2_score
        r2 = r2_score(all_test_labels, all_pred_labels)
        
        print("\n整体预测性能:")
        print(f"预测准确率: {accuracy*100:.2f}%")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        
        print("\n类别精确度:")
        print(f"卖出信号精确度: {class_precision[0]:.4f}")
        print(f"持有信号精确度: {class_precision[1]:.4f}")
        print(f"买入信号精确度: {class_precision[2]:.4f}")
        
        print("\n类别召回率:")
        print(f"卖出信号召回率: {class_recall[0]:.4f}")
        print(f"持有信号召回率: {class_recall[1]:.4f}")
        print(f"买入信号召回率: {class_recall[2]:.4f}")
        
        print("\n预测分布特征:")
        print(f"均值: {pred_mean:.4f}")
        print(f"标准差: {pred_std:.4f}")
        print(f"偏度: {pred_skew:.4f}")
        print(f"峰度: {pred_kurtosis:.4f}")
        
        print(f"\n{confidence*100}%置信区间: [{interval[0]:.4f}, {interval[1]:.4f}]")
        
        print("\n额外统计指标:")
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
            'Sell_Precision': class_precision[0],
            'Hold_Precision': class_precision[1],
            'Buy_Precision': class_precision[2],
            'Sell_Recall': class_recall[0],
            'Hold_Recall': class_recall[1],
            'Buy_Recall': class_recall[2],
            'Pred_Mean': pred_mean,
            'Pred_Std': pred_std,
            'Pred_Skewness': pred_skew,
            'Pred_Kurtosis': pred_kurtosis,
            'CI_Lower': interval[0],
            'CI_Upper': interval[1],
            'Markov_params': f"Regimes={best_params[0]}, Order={best_params[1]}",
            'AIC': model_results.aic
        }
        
    except Exception as e:
        print(f"处理 {file_path} 时发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

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


def run(data_dir, output_csv="markov_trading_results.csv"):
    results = {}
    for file_path, initial_money, train_ratio, asset_name in _jobs(data_dir):
        print("\n" + "=" * 50)
        print("处理文件: %s" % file_path)
        close_prices, signals, train_size, prediction_metrics = markov_trading(
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

    parser = argparse.ArgumentParser(description="Markov switching baseline in Table 3 and Table 4")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="markov_trading_results.csv")
    args = parser.parse_args()
    print(run(args.data_dir, args.output))