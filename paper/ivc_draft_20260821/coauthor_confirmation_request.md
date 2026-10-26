# 共著者確認依頼（IVC 投稿）

2026-09-18 作成。共著者には会議論文との差分の共有と、著者リスト・所属の確認のみを依頼する。利益相反・CRediT・データ／倫理・生成 AI の各記載は `main.tex` に記入済みで、責任著者が投稿時にまとめて処理する。

---

件名：【ご確認のお願い】CanonFuse3D ジャーナル論文（Image and Vision Computing）の投稿について（〆切：[日付]）

北原先生、内藤先生、木村様、謝先生、王さん

お世話になっております。筑波大学の陳です。

MMSports 2026 論文のジャーナル拡張版を Elsevier の Image and Vision Computing に投稿する準備が整いました。会議論文との差分を下記にまとめましたので、あわせて著者リストと所属のご確認をいただけますでしょうか。現在の原稿（main.pdf：43 ページ、supplementary.pdf：26 ページ）を添付いたします。

**会議論文（MMSports 2026）との差分（概要）**

会議論文から引き継いでいるのは、CanonFuse3D のアーキテクチャと学習目的、問題設定、および 2 つの定量評価データセット（Unity 合成データ、Ski-PTZ-Pose）です。数値表・図・定性例は会議論文から一切再利用しておらず、すべてジャーナル用のプロトコル（Unity 全 64,440 系列、Ski-PTZ-Pose 固定 30 系列）で再生成しています。ジャーナル版で新たに加えた内容は次のとおりです。

- 主比較の拡張：両データセットで位置誤差・Procrustes 整列誤差・加速度誤差を報告し、平均化系ベースラインに加え、学習型 TCN、SmoothNet、STRIDE / DeciWatch / MetaPose の適応版、キャリブレーション付き三角測量群と比較。正規化前の native 出力での独立比較も追加。
- ゲート・残差補正のアブレーション（再学習 3 構成）と、凍結モデルに対するゲート診断・介入実験。
- 頑健性の系統的評価：カメラ角度差（6 区間、信頼区間・検定・効果量つき）、時間オフセット（サブフレーム含む）とサンプリングレート誤差、45 条件の姿勢欠損、SAM3D を経由した 18 条件の画像レベル遮蔽、同一フレームから得た 2D / 3D 推定の対応比較とキャリブレーション誤差感度。
- フロントエンド汎化：SAM3D / VideoPose3D / PoseFormer / MotionBERT 間の凍結転移と、7 通りの軽量適応（8×4 の完全行列で評価）。
- 多視点拡張：1〜4 台のペアワイズ合成による精度と計算コストの同時計測、融合モジュールの効率監査、自動時間整合の診断。

会議論文との関係はカバーレターで編集部に説明し、採録済みの会議原稿も投稿時に添付します。

**著者リストと所属（タイトルページ）**

Kaixu Chen (a, 責任著者), Qingyu Wang (b), Chun Xie (a), Eiichi Naito (c), Toshitaka Kimura (d), Itaru Kitahara (a)

Elsevier では投稿時の著者リストが確定版として扱われ、以降の追加・削除・順序変更には編集者の承認と全著者の書面同意が必要になりますので、この時点でご確認をお願いいたします。

- (a) Center for Computational Sciences, University of Tsukuba, 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577, Japan
- (b) Degree Programs in Systems and Information Engineering, Graduate School of Science and Technology, University of Tsukuba, 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8573, Japan
- (c) Center for Information and Neural Networks (CiNet), Advanced ICT Research Institute, NICT, 1-4 Yamadaoka, Suita, Osaka 565-0871, Japan
  ※内藤先生：所属表記がこれでよいかご確認をお願いいたします。
- (d) NTT Communication Science Laboratories, NTT, Inc., 3-1 Morinosato-Wakamiya, Atsugi, Kanagawa 243-0198, Japan
  ※木村様：住所のご確認をお願いいたします（旧稿では Tokyo となっておりました）。

お忙しいところ恐れ入りますが、[日付] までにご返信いただけますと幸いです。
どうぞよろしくお願いいたします。

陳 凱旭（Kaixu Chen）
筑波大学 計算科学研究センター
chenkaixusan@gmail.com

---

## 只有通讯作者能做的事

- Editorial Manager 里填通讯作者 ORCID。ORCID 检索到 `0009-0003-8698-8899`（"kaixu chen", University of Tsukuba），请确认是否为本人。
- 在 https://declarations.elsevier.com/ 生成 Declaration of Interest 的 .docx（全员"无利益冲突"），原样上传。
- `main.tex` 里带 `% CONFIRM` 注释的段落（利益冲突、CRediT、数据可用性、伦理、生成 AI 声明）由通讯作者自行核定后删除注释。
- 从 ACM TAPS 获取 MMSports 2026 录用稿 PDF，放入本目录。
