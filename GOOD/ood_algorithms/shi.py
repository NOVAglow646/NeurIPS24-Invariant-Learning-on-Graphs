import pandas as pd

# 加载提供的Excel文件
file_path = '/path/to/your/excel/file.xlsx'  # 替换为实际的文件路径
data = pd.read_excel(file_path)

from scipy.stats import chi2_contingency

# 过滤掉'Q7_Flood_house'为999的行
filtered_data = data[data['Q7_Flood_house'] != 999]

# 创建一个新的列联表，不包括999值
new_contingency_table = pd.crosstab(filtered_data['Q4_House_ownership'], filtered_data['Q7_Flood_house'])

# 执行卡方检验
chi2, p, dof, expected = chi2_contingency(new_contingency_table)

# 打印卡方统计量和P值
print("Chi-Squared Statistic:", chi2)
print("P-Value:", p)
