# UJI cross-floor DG/DA, V1

## Research question

Within one building, can a coordinate regressor trained on several historical floors locate scans from a wholly unseen floor? GGA tests whether its early source-domain gradient alignment improves this *domain generalization* over matched ERM. Set Transformer separately tests whether representing each scan as an unordered set of detected (AP ID, RSSI) pairs helps. Target-floor adaptation is a separate 10-position, 3-scans-per-position experiment, capped at 50 supervised update steps.

## Domains and target permissions

One BuildingID–Floor pair is one domain. For target floor t, train on `trainingData.csv` rows from every other floor of the **same building**. These source floors have globally comparable horizontal coordinates. No target-floor `trainingData.csv` row or `validationData.csv` row enters DG model fitting or GGA annealing. `validationData.csv` rows of t are the **primary final test**, as the UJI release intended. Target `trainingData.csv` rows are an additional spatial DG diagnostic only after fitting; they are not used to choose a checkpoint. The official validation file is never used for hyperparameters, early stopping, or target adaptation.

For DA only, select 10 distinct coordinate positions from target `trainingData.csv` with at least 3 scans per position, then 3 scans per position, with a fixed seed. Fine-tune the already fitted DG model's output head for exactly 50 steps using only those 30 rows. Report its official target-floor validation error at steps 0, 1, 5, 10, 20, 50; these are reporting points, not checkpoint candidates. DG and DA numbers must be reported separately. The same support row IDs and test rows apply to all four methods for a target/seed.

## Inputs, labels and loss

Only WAP001–WAP520 RSSI enter the model. `100` means undetected. Dense MLP input replaces it by −110 dBm, maps RSSI to `[0,1]`. Set encoder sees only detected APs as an unordered set of (AP ID, normalized RSSI); a null token represents an empty scan. Building/Floor select domains, not network features. USERID, PHONEID, TIMESTAMP are not model inputs; their joint shifts remain real confounders and are described, not claimed removed.

The AP vocabulary is fitted on source floors only. Any AP absent from all source scans is treated as undetected in target support and tests for **both** encoders. This avoids feeding randomly untrained AP-specific weights at test time. The proportion of AP observations excluded is reported, since this projection also discards target information.

Output is (LONGITUDE,LATITUDE) on the known target building/floor. Source-only coordinate origin is the mean source coordinate for a target run; labels are `(xy-origin)/100`, the source regression loss is mean per-scan squared 2-D norm in that scale. Test metric is mean Euclidean horizontal error in metres per floor. Report floors separately and macro-average floors equally; include sample counts and three-seed variation. No building/floor classification score is part of V1.

## GGA transfer from the original classification code

Official GGA (updated arXiv v7, official GitHub commit in provenance) perturbs only the feature extractor in the early step window. For each candidate, calculate each source floor's regression-loss gradient over **all model parameters** on its current minibatch; score the minimum pairwise cosine similarity. Accept a candidate only if this minimum improves and the combined minibatch loss increases by less than 10% relative to the best accepted loss. The original code allows +0.1 absolute cross-entropy; since our scaled regression loss is about 0.01 in this window, an absolute +0.1 would make the gate effectively inert. The relative gate preserves the intended scale of the restriction under arbitrary coordinate units. Restore the best candidate weights and take the ordinary Adam step. Preserve the official 100–200 annealing window, 250 perturbation candidates, uniform ±1e−5 perturbations, and 5000 total source steps. Cross-entropy is replaced by the scaled coordinate regression loss; ImageNet pretrained ResNet is replaced by an RSSI encoder. These are task-driven adaptations, so results are a GGA-on-UJI investigation, not a numerical reproduction of CVPR image classification tables.

Matched ERM uses identical RSSI encoder, train minibatches, Adam settings and 5000 source steps without GGA search. MLP and Set encoders are run as a 2×2 comparison. No target test information controls training.

## Temporal and nuisance interpretation

UJI's official validation was collected later and with a different user/phone composition. A difference between target `trainingData.csv` and target `validationData.csv` therefore combines floor novelty, time drift, device/user sampling and possible location coverage changes. It cannot be attributed to spatial shift alone. Domain gradients are grouped by floor, never by time or phone. The older target-train diagnostic and later official test are always shown side by side to expose this mixture.
