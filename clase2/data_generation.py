import numpy as np
import pandas as pd

N = 15000000
id = np.arange(1, N + 1)
sexo = ['M', 'F']
region = ['Lima', 'Provincia']
comportamiento_financiero = ['Sin deudas', 'Endeudado', 'Sobreendeudado']

distribucion_sexo = np.random.choice(sexo, size=N, p=[0.5, 0.5])

distribucion_edad = np.random.normal(loc=42, scale=12, size=N)
distribucion_edad = np.clip(distribucion_edad, 18, 65).astype(int)

distribucion_region = np.random.choice(region, size=N, p=[0.7, 0.3])

# Generate income with multiple conditional effects:
# 1. Gender gap: Women earn 20% less than men
# 2. Region gap: Lima earns 30% more than Provincia
# 3. Age effect: Income increases with age

# Create masks
mask_men = distribucion_sexo == 'M'
mask_women = distribucion_sexo == 'F'
mask_lima = distribucion_region == 'Lima'
mask_provincia = distribucion_region == 'Provincia'

# Base parameters
base_log_income = np.log(1500)  # Base for young person in Provincia
sigma = 0.4

# Calculate log-income adjustments using a linear model in log-space
# log(income) = base + gender_effect + region_effect + age_effect + noise

log_income = np.full(N, base_log_income)

# Gender effect: women earn 20% less (multiply by 0.8 = add log(0.8) in log-space)
gender_effect = np.where(mask_women, np.log(0.8), 0)

# Region effect: Lima earns 30% more (multiply by 1.3 = add log(1.3) in log-space)
region_effect = np.where(mask_lima, np.log(1.3), 0)

# Age effect: income increases ~2% per year of age above 18
# At age 18: multiplier = 1.0, at age 65: multiplier = 1.0 * 1.02^47 ≈ 2.5x
age_coefficient = 0.02  # 2% increase per year
age_effect = age_coefficient * (distribucion_edad - 18)

# Combine all effects
log_income = log_income + gender_effect + region_effect + age_effect

# Add random noise (lognormal)
noise = np.random.normal(0, sigma, size=N)
log_income = log_income + noise

# Convert to actual income
distribucion_ingreso = np.exp(log_income)
distribucion_ingreso = np.clip(distribucion_ingreso, 1130, 50000).astype(int)

# Generate financial behavior conditionally based on sex
# Men have worse financial behavior (more 'Endeudado' and 'Sobreendeudado')
distribucion_comportamiento_financiero = np.empty(N, dtype=object)

# Men: less 'Sin deudas', more debt problems
prob_men = [0.40, 0.35, 0.25]  # Sin deudas, Endeudado, Sobreendeudado
# Women: more 'Sin deudas', fewer debt problems
prob_women = [0.60, 0.25, 0.15]  # Sin deudas, Endeudado, Sobreendeudado

distribucion_comportamiento_financiero[mask_men] = np.random.choice(
    comportamiento_financiero, size=mask_men.sum(), p=prob_men
)
distribucion_comportamiento_financiero[mask_women] = np.random.choice(
    comportamiento_financiero, size=mask_women.sum(), p=prob_women
)

dataset = pd.DataFrame({
    'id': id,
    'sexo': distribucion_sexo,
    'edad': distribucion_edad,
    'region': distribucion_region,
    'ingreso_mensual': distribucion_ingreso,
    'comportamiento_financiero': distribucion_comportamiento_financiero
})

print(dataset.head())

dataset.to_csv('clase2/data.csv', index=False)