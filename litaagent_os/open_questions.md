最后更新：2026-01-09 11:36

已决事项：
1. 模型工件目录与命名统一为 `assets/models/accept/` 与 `assets/models/breach/`，文件名固定 `model.bin` + `model_meta.json`，可选 `calibration.json`；目录与模型类型分离以保证互换一致。
2. `predict_accept_strength` 默认强度固定为常数，默认 `s=8`，推荐范围 `[2, 20]`。
3. `buffer(t)` 固定为单调上升函数，采用 `buffer(t)=b_min+(b_max-b_min)*t^gamma`，`t=round_rel`；默认 `b_min=0.05`、`b_max=0.35`、`gamma=2`。
4. `m_i(o_i)` 采用 trading_price 近似（在线可得）并保留单调兜底：卖方 `m=(p-trading_price)/max(1e-6,trading_price)`，买方 `m=(trading_price-p)/max(1e-6,trading_price)`，再乘数量得到 `m_i=q*m`。
5. counter_all 的子集选择：若是“接受对方 offer”，视为 `P_sign=1`，因此 `\tilde q=q*LCB(P_fulfill)`；若是“我方发出 counter_offer”，仍用 `P_eff=LCB(P_sign)*LCB(P_fulfill)`。
6. terminal negative 只在“对手 END / timeout / failed 且无 agreement”时启用，并使用 `w_end` 作为软负权重；我方主动 END 不计入。

未决事项：
- 暂无
