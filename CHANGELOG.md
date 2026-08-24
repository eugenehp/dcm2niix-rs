# Changelog

## Unreleased

## [0.1.0] — 2026-08-22

First public release of the pure-Rust `dcm2niix` port. CLI-compatible conversion for classic MR/CT/PET, mosaics, enhanced multi-frame, MRS, PAR/REC, and BIDS sidecars. Passes the `dcm_qa` / `dcm_qa_nih` / `dcm_qa_uih` parity gate against upstream reference outputs.

### Added
- BIDS `EchoNumber`; `DeidentificationMethod` / `DeidentificationMethodCodeSequence` (issue 877)
- BIDS `AcquisitionDateTime` when `-ba` is not full anonymize
- BIDS `ReferringPhysicianName` / `PatientAge` (years); `PulseSequenceName` + SequenceName promotion
- BIDS `FmriExternalInfo` field (empty until AscConvEnd parse matches C++; C++ read still commented)
- PET `ImageDecayCorrected` / `ImageDecayCorrectionTime`
- MRS: `WaterSuppressed`, `TransmitCoilName`, `MRSpectroscopyAcquisitionType`, ResonantNucleus / SpectrometerFrequency
- Issue 560: TR reconciliation for PET and 2D-stacked MR (`force_onset`, non-GE); skip enhanced MF / mosaic (UIH-safe)
- MRS: `NumberOfSpectralPoints`, `NumberOfTransients`, `AcquisitionVoxelSize`; BIDS-MRS `ScanningSequence` (SVS/MRSI)
- MRS: BIDS / NIfTI-MRS `VOI` 4×4 (CSA + VolumeLocalizationSequence; `mrsVoiMatrix`)
- AcquisitionTime: prefer last `(0008,0032)` when tag repeats (C++ last-wins)
- MGZ fail-closed: remove partial `.mgh`/`.mgz` on write error
- CSA `AveragesDouble`; BIDS `ConversionComments` from `-c`; CT `ExposureTime` / `XRayTubeCurrent` / `XRayExposure`
- Warn when PixelSpacing varies within a series (issue 1009)
- MRS `(0018,9052)` SpectralWidth + `(0018,9093)` NumberOfKSpaceTrajectories; GE XML protocol-block warning
- Public `(0018,9073)` AcquisitionDuration for non-UIH BIDS
- ASL `RepetitionTimePreparation` (ms, C++ quirk) for pCASL sequences; public ASL `LabelingOrientation` / `VascularCrushing` / `VascularCrushingVENC`
- PET single-volume `FrameTimesStart` BIDS fallback when onset array empty (issue 983)
- Issue 690/777: zero GE `NumberOfDiffusionDirectionGE` for non-DTI (`VasCollapseFlag` / series diffusion type)
- GE early HyperBand SliceTiming warning; ASL `M0Type` undetermined warning
- BIDS `UsePhilipsFloatNotDisplayScaling` follows CLI `-p`; `CompressedSensingFactor` from GE `(0043,10B7)`
- BIDS `ImageTypeText` array; `RawImage`; `DeepLearning` / `DeepLearningDetails`
- `FrequencyEncodingSteps`, `PhaseNumber`, `VariableFlipAngleFlag`, `TriggerDelayTime`, `ParallelAcquisitionTechnique`
- ImageType appends PHASE/REAL/IMAGINARY/FIELDMAPHZ (issue 881) while keeping mosaic/GE/UIH MAGNITUDE
- PartialFourierDirection + Philips PartialFourierEnabled / PhaseEncodingStepsNoPartialFourier (issue 377)
- GE fieldmapHz EchoTime1/EchoTime2 (issue 617); maxEchoNumGE warning (issue 359)
- Philips RWV / Rescale / ScaleSlope BIDS fields
- GE epi2 BIDS: `NumberOfDiffusionDirectionGE` / `NumberOfDiffusionT2GE` / `TensorFileNumberGE`
- PET issue 802: `RandomsCorrectionMethod` / `ScatterCorrectionMethod` / `ReconMethodName` + subset/iteration params
- Unknown-manufacturer (0008,0070) conversion warning
- GE diffusion ALLTR SliceTiming: same within-TR epi2 pattern as OFF/2TR/3TR (issue 635; ahead of C++)
- GE diffusion 2TR/3TR SliceTiming (within-TR EPI pattern; ALLTR previously unsupported); cycling-mode DICOM detection
- ASL BIDS `BackgroundSuppression` from CSA `sAsl.ulSuppressionMode` (1=off, ≥2=on)
- Issue 642: `Quadruped` BIDS flag + PatientPosition / QUADRUPED warnings; DICOMANON warning (issue 383)
- Encapsulated JPEG fragment size≤8 fail-closed (mirrors upstream J2K guard)
- Philips `_Raw` / `_PS` SOP postfixes; PET ConvolutionKernel / ReconFilterType+Size / SeriesTime+ScanStart
- `_fieldmaphz` filename postfix; PET `ScatterFraction` as BIDS array; once-only `%h`/`%H` warning
- NRRD fail-closed partial-file cleanup
- Siemens 3D EPI volume TR (issue #1024): MultiEchoShots, RepetitionTimeExcitation, PE-axis gate
- Multi-DICOM MRSI CSI stacking (slice-axis concat beyond upstream stub)
- GE pepolar (`epi_pepolar` + userData12 / volume polarity + extra Y-flip)
- Siemens RF-off `_noRF` from `(0021,1175)` ImageTypeText `NOISE` token (issue #1025)
- Optional `gpu` feature (wgpu via rlx) for large volume flips; `DCM2NIIX_RLX_DEVICE`
- JNIfTI / BNIfTI NeuroJSON export; NRRD multi-volume DWMRI gradients
- ReproIn `.reproin_provenance.tsv`; Philips multi-dynamic SVS unpack
- Parallel slice/mosaic decode (rayon); fused Y/Z flips
- CI workflows (`ci.yml` build/test, `parity.yml` dcm_qa gate); `.gitignore`

### Changed
- Default voxel flips use a tight in-place CPU path (rlx/wgpu reserved for large volumes)
- `rlx-tensor` from crates.io `0.2.14` (publishable; was path-only `0.2.15`)
- Ortho reorient (`nii_setOrtho`) uses the same rlx flip+transpose path (CPU gather below ~8 MiB)
- Faster flips (whole-row / whole-slice `swap_with_slice`); parallel mosaic demosaic, slice pack, and 4D ortho
- MRS `VOI` uses full-precision IOP (`voi_orient`) instead of `snap_f32` sform orientation
- Parallel DICOM header scan; series grouping pre-buckets by UID; parallel 16-bit pack / `-l` scale
- Fast Part-10 `DICM` sniff for file discovery (avoid double-parse); parallel multi-series convert; parallel Y-flip planes
- Bounded job pool (`DCM2NIIX_JOBS`, default ≤8): decode/header/multi-series/`-a y`; full-file prefetch open (&lt;64 MiB)
- Gzip: default level 1; `-z y` auto-uses piped pigz when found
- Move-based series grouping (no `DicomImage` clone); 1 MiB NIfTI write buffers; overlap BIDS JSON with NIfTI write/compress
- Fast monochrome 8/16-bit raw→`f32` decode (parallel); `ModalityLut::None` fallback drops `f64` detour; enable `dicom-pixeldata` rayon
- `open` uses mmap (&lt;64 MiB); enhanced MF decodes from the already-opened object (no second parse)
- Owned voxel buffers through write path (no forced copy); MF truncate in place; native-endian NIfTI writes from slice
- Classic series: move into `SliceMeta` (one header clone); overlay ROI path uses `&DicomImage` refs
- `volume_representatives` + ref-based onset/TR/DTI/NRRD (no per-volume `DicomImage` clone on 4D stack)
- Header scan stops before Pixel Data (`read_until`); discovery fallback uses same path
- Mosaic series: move into sort (no `to_vec`); `split_at_mut` slice timing; drop path clones on decode
- MRS SVS/MRSI: move into sort (no `to_vec`); drop `_mrsref` companion clone
- `spectroscopy_data_from_object`; parallel MRS FID load (rayon)
- Series-local `prefetch_mmaps` for parallel slice/MRS decode (reuse mmap, skip per-thread `open`)
- Convert-scoped mmap cache: one prefetch for header scan + all series decode/MRS
- Fused `warmup_convert_cache`: single parallel pass mmap + header parse (no duplicate file open)
- Enhanced multi-frame decode reuses convert mmap cache via `open_prefetched`
- DTI `_ADC` companion: write before truncate (no full-volume `bytes.clone()`)
- `looks_like_dicom` fallback uses header-only mmap parse (≤64 MiB)
- Slice stack sort: permute by index (no `DicomImage` clone for instance reorder)
- Classic stack decode: write slices directly into one volume (skip `Vec<Vec<f32>>` + `pack_slices`)
- BIDS / MRS / PET / GE slice-timing coverage aligned with upstream for `dcm_qa*`

### Parity
- `dcm-parity --all` passes on `dcm_qa`, `dcm_qa_nih`, `dcm_qa_uih`
