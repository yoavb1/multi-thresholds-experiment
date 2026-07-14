import numpy as np
import pandas as pd


def generate_labels(n_items, ps, rng):
    n_signal = int(n_items * ps)
    n_noise = n_items - n_signal

    labels = np.array(["signal"] * n_signal + ["noise"] * n_noise)

    rng.shuffle(labels)

    return labels


def force_exact_mean_and_sd(values, target_mean, target_sd):

    values = np.array(values)

    values = values - values.mean()
    current_sd = values.std(ddof=1)

    if current_sd == 0:
        raise ValueError("Cannot force SD because all sampled values are identical.")

    values = values / current_sd
    values = values * target_sd
    values = values + target_mean

    return values


def sample_sdt_values(labels, dprime, sd, rng):
    

    labels = np.array(labels)
    values = np.zeros(len(labels))

    mu_signal = dprime / 2
    mu_noise = -dprime / 2

    signal_indices = np.where(labels == "signal")[0]
    noise_indices = np.where(labels == "noise")[0]

    signal_values = rng.normal(
        loc=mu_signal,
        scale=sd,
        size=len(signal_indices)
    )

    signal_values = force_exact_mean_and_sd(
        values=signal_values,
        target_mean=mu_signal,
        target_sd=sd
    )

    noise_values = rng.normal(
        loc=mu_noise,
        scale=sd,
        size=len(noise_indices)
    )

    noise_values = force_exact_mean_and_sd(
        values=noise_values,
        target_mean=mu_noise,
        target_sd=sd
    )

    values[signal_indices] = signal_values
    values[noise_indices] = noise_values

    return values


def generate_ai_classification(x_ai, blow, bhigh):

    if x_ai < blow:
        return "noise"
    elif x_ai <= bhigh:
        return "uncertain"
    else:
        return "signal"


def generate_one_block(
    condition_id,
    block,
    block_n_items,
    dprime_ai,
    dprime_human,
    thresholds_distance,
    ps,
    sd,
    seed
):

    rng = np.random.default_rng(seed)

    blow = -thresholds_distance / 2
    bhigh = thresholds_distance / 2

    labels = generate_labels(
        n_items=block_n_items,
        ps=ps,
        rng=rng
    )

    x_ai = sample_sdt_values(
        labels=labels,
        dprime=dprime_ai,
        sd=sd,
        rng=rng
    )

    x_human = sample_sdt_values(
        labels=labels,
        dprime=dprime_human,
        sd=sd,
        rng=rng
    )

    ai_classifications = [
        generate_ai_classification(value, blow, bhigh)
        for value in x_ai
    ]

    df_block = pd.DataFrame({
        "condition_id": condition_id,
        "block": block,
        "ps": ps,
        "dprime_ai": dprime_ai,
        "dprime_human": dprime_human,
        "thresholds_distance": thresholds_distance,
        "blow": blow,
        "bhigh": bhigh,
        "true_label": labels,
        "x_ai": x_ai,
        "x_human": x_human,
        "ai_classification": ai_classifications
    })

    return df_block


def generate_one_condition(
    condition_id,
    dprime_ai,
    dprime_human,
    thresholds_distance,
    ps,
    sd,
    seed
):


    block_sizes = {
        1: 10,
        2: 10,
        3: 100
    }

    blocks = []

    for block, block_n_items in block_sizes.items():
        df_block = generate_one_block(
            condition_id=condition_id,
            block=block,
            block_n_items=block_n_items,
            dprime_ai=dprime_ai,
            dprime_human=dprime_human,
            thresholds_distance=thresholds_distance,
            ps=ps,
            sd=sd,
            seed=seed + block
        )

        blocks.append(df_block)

    df_condition = pd.concat(blocks, ignore_index=True)

    df_condition["item_id"] = np.arange(1, len(df_condition) + 1)

    df_condition = df_condition[
        [
            "condition_id",
            "item_id",
            "block",
            "ps",
            "dprime_ai",
            "dprime_human",
            "thresholds_distance",
            "blow",
            "bhigh",
            "true_label",
            "x_ai",
            "x_human",
            "ai_classification"
        ]
    ]

    return df_condition


def generate_experiment_sets(
    dprime_ai_low,
    dprime_ai_high,
    dprime_human_low,
    dprime_human_high,
    threshold_distance_small,
    threshold_distance_large,
    ps,
    sd=1,
    output_file="experiment_data.csv",
    seed=123
):


    conditions = []
    condition_id = 1

    for dprime_ai in [dprime_ai_low, dprime_ai_high]:
        for dprime_human in [dprime_human_low, dprime_human_high]:
            for thresholds_distance in [threshold_distance_small, threshold_distance_large]:

                conditions.append({
                    "condition_id": condition_id,
                    "dprime_ai": dprime_ai,
                    "dprime_human": dprime_human,
                    "thresholds_distance": thresholds_distance
                })

                condition_id += 1

    all_conditions = []

    for condition in conditions:
        df_condition = generate_one_condition(
            condition_id=condition["condition_id"],
            dprime_ai=condition["dprime_ai"],
            dprime_human=condition["dprime_human"],
            thresholds_distance=condition["thresholds_distance"],
            ps=ps,
            sd=sd,
            seed=seed + condition["condition_id"] * 100
        )

        all_conditions.append(df_condition)

    final_df = pd.concat(all_conditions, ignore_index=True)

    # final_df = final_df[final_df['dprime_human'] == 1.5]

    # final_df.to_csv(output_file, index=False)
    print(final_df.head())

    return final_df

def check_data(df):
    print(df[['blow', 'bhigh', 'ai_classification','item_id']].groupby(['blow', 'bhigh', 'ai_classification']).count())

if __name__ == "__main__":
    df = generate_experiment_sets(
            dprime_ai_low=1,
            dprime_ai_high=2,
            dprime_human_low=1.5,
            dprime_human_high=2,
            threshold_distance_small=2,
            threshold_distance_large=3,
            ps=0.5,
            sd=1,
            output_file="data/data.csv",
            seed=123
        )
    print(df.shape[0])
    check_data(df)