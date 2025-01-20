import pandas as pd

# 提取的数据结构化形式：从提供的文本手动解析
data = {
    "dkt+_ASSIST2009": {
        "FL": {
            "Best loss": (99, 0.5115785002708435),
            "Best accuracy": (58, 0.7716244085891059),
            "Best AUC": (70, 0.8151655886922364)
        },
        "Training": {
            "Best loss": (98, 0.57602644),
            "Best accuracy": (42, 0.7712751515882832),
            "Best AUC": (49, 0.8188941607763446)
        }
    },
    "dkt+_ASSIST2015": {
        "FL": {
            "Best loss": (63, 0.5463559031486511),
            "Best accuracy": (29, 0.7526886466287093),
            "Best AUC": (29, 0.7227867689860388)
        },
        "Training": {
            "Best loss": (99, 0.60216445),
            "Best accuracy": (4, 0.7560150144606486),
            "Best AUC": (30, 0.7219095906737993)
        }
    },
    "dkt_ASSIST2009": {
        "FL": {
            "Best loss": (83, 0.4989196956157684),
            "Best accuracy": (33, 0.7715556108329965),
            "Best AUC": (25, 0.8193808612687349)
        },
        "Training": {
            "Best loss": (98, 0.5252925),
            "Best accuracy": (23, 0.770852816072883),
            "Best AUC": (24, 0.818047709126044)
        }
    },
    "dkt_ASSIST2015": {
        "FL": {
            "Best loss": (17, 0.5401346683502197),
            "Best accuracy": (5, 0.7517101211873018),
            "Best AUC": (6, 0.72656705594822)
        },
        "Training": {
            "Best loss": (99, 0.55659044),
            "Best accuracy": (10, 0.7585225524583102),
            "Best AUC": (8, 0.7294761181117312)
        }
    },
    "dkvmn_ASSIST2009": {
        "FL": {
            "Best loss": (99, 0.486895889043808),
            "Best accuracy": (94, 0.7651064883605745),
            "Best AUC": (99, 0.8022092027360677)
        },
        "Training": {
            "Best loss": (99, 0.47234833),
            "Best accuracy": (93, 0.7652417870825666),
            "Best AUC": (98, 0.8069954659243057)
        }
    },
    "dkvmn_ASSIST2015": {
        "FL": {
            "Best loss": (89, 0.5162424445152283),
            "Best accuracy": (91, 0.7497300767457469),
            "Best AUC": (93, 0.7252882268701358)
        },
        "Training": {
            "Best loss": (98, 0.5098828),
            "Best accuracy": (87, 0.7552304473570857),
            "Best AUC": (72, 0.7268855699876311)
        }
    },
    "kqn_ASSIST2009": {
        "FL": {
            "Best loss": (97, 0.5335273146629333),
            "Best accuracy": (97, 0.7539407920030757),
            "Best AUC": (97, 0.7835849675647328)
        },
        "Training": {
            "Best loss": (99, 0.49848765),
            "Best accuracy": (72, 0.7554074029382485),
            "Best AUC": (98, 0.7878604481098507)
        }
    },
    "kqn_ASSIST2015": {
        "FL": {
            "Best loss": (95, 0.583090603351593),
            "Best accuracy": (71, 0.7429629629629629),
            "Best AUC": (79, 0.7155211828974475)
        },
        "Training": {
            "Best loss": (93, 0.5155137),
            "Best accuracy": (96, 0.7537382314934465),
            "Best AUC": (99, 0.7202844349596165)
        }
    },
    "sakt_ASSIST2009": {
        "FL": {
            "Best loss": (36, 0.5032500624656677),
            "Best accuracy": (34, 0.7579496311371152),
            "Best AUC": (36, 0.7960886173030629)
        },
        "Training": {
            "Best loss": (99, 0.44527718),
            "Best accuracy": (34, 0.7655434553078524),
            "Best AUC": (24, 0.8068400119290576)
        }
    },
    "sakt_ASSIST2015": {
        "FL": {
            "Best loss": (22, 0.5113258361816406),
            "Best accuracy": (22, 0.7565644208390523),
            "Best AUC": (18, 0.7241092199140221)
        },
        "Training": {
            "Best loss": (99, 0.4786156),
            "Best accuracy": (12, 0.7595378745923328),
            "Best AUC": (12, 0.7304036752247927)
        }
    }
}

# 创建数据框的函数
def create_model_df(model_data):
    rows = []
    for model_name, versions in model_data.items():
        for version, metrics in versions.items():
            row = {
                "Model": model_name,
                "Version": version,
                "Best loss (Epoch/Round)": f"Round {metrics['Best loss'][0]}" if version == "FL" else f"Epoch {metrics['Best loss'][0]}",
                "Loss Value": metrics["Best loss"][1],
                "Best accuracy (Epoch/Round)": f"Round {metrics['Best accuracy'][0]}" if version == "FL" else f"Epoch {metrics['Best accuracy'][0]}",
                "Accuracy Value": f"{metrics['Best accuracy'][1] * 100:.2f}%",  # 转换为百分比并保留两位小数
                "Best AUC (Epoch/Round)": f"Round {metrics['Best AUC'][0]}" if version == "FL" else f"Epoch {metrics['Best AUC'][0]}",
                "AUC Value": f"{metrics['Best AUC'][1] * 100:.2f}%"  # 转换为百分比并保留两位小数
            }
            rows.append(row)
    return pd.DataFrame(rows)

# 创建ASSIST2009和ASSIST2015的数据框
df_ASSIST2009 = create_model_df({key: value for key, value in data.items() if "ASSIST2009" in key})
df_ASSIST2015 = create_model_df({key: value for key, value in data.items() if "ASSIST2015" in key})

# 将数据写入Excel文件
with pd.ExcelWriter('model_results.xlsx', engine='xlsxwriter') as writer:
    df_ASSIST2009.to_excel(writer, sheet_name="ASSIST2009", index=False)
    df_ASSIST2015.to_excel(writer, sheet_name="ASSIST2015", index=False)

print("Excel文件 'model_results.xlsx' 已成功创建。")
