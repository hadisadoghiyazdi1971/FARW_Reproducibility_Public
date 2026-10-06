clc; clearvars; close all;

%% =======================================================================
%  Feature Extraction Script — v8 (conservative corrected schema)
%  - 40 test conditions × 4 repetitions (from the new layout)
%  - 2-class labels: Stable(1), Chatter(0) — Transition class removed
%  - Fixed 350 windows per signal (Short signals extracted as available)
%  - Raw signal (no noise removal)
%  - v4: added wRCMDE (Yang, Guo & Sun, 2022) and MPE (Liu et al., 2021)
%        features 38-42. All original v3 code/logic is unchanged below;
%        new code is clearly marked and appended only.
%  - v5: added CE (Liu et al., 2021, Eq. 9) and three additional MPE
%        scale factors (s = 1, 2, 3), features 43-46. All v3/v4
%        code/logic is unchanged below; new code is clearly marked and
%        appended only.
%  - v6: SUPERSEDES the multi-scale part of v4/v5. Per paper, wRCMDE is
%        the only one of the two that is genuinely a multi-scale feature
%        set in its source paper (Yang et al. use s = 1..4 jointly as 4
%        inputs to their SVM); MPE in Liu et al. is a single best-scale
%        index (s = 4) chosen from a sweep, not a multi-scale vector. To
%        match the 40-feature table (22 time + 14 freq + CE + WPEE + one
%        MPE + one wRCMDE = 40), both are now reduced to ONE feature
%        each: MPE_s1/s2/s3 (added in v5) are removed, keeping only the
%        paper-justified MPE at s = 4; wRCMDE_s1/s2/s3/s4 (added in v4)
%        are collapsed to a single chosen scale (see parameter block
%        below for which one and why this is a judgment call, not a
%        literal reproduction of Yang et al.). CE is unchanged. Total
%        feature count: 40.
%  - v7: corrected dimensionally invalid time/frequency formulas; adopted
%        explicit power- and magnitude-spectrum conventions; corrected
%        short-record window counting; and introduced an auditable 40-field
%        schema. Outputs from v7 are not numerically interchangeable with v6.
%  - v8: keeps the window-count rule unchanged; retains source definitions
%        where methodologically sound while using standard moment coefficients
%        for skewness/kurtosis; corrects the invalid frequency moments; and
%        replaces three
%        algebraically redundant features with Hjorth complexity, power-
%        spectral skewness, and power-spectral kurtosis. No eps is added to
%        any physical denominator. Outputs are not interchangeable with v7.
%  - Output: one .mat file per signal containing SigData, FeatureNames, and
%        FeatureSchemaVersion.
%% =======================================================================

% -----------------------------------------------------------------------
% Signal list: {filename, label_value, label_string}
%   1 = Stable (S)
%   0 = Chatter (U)
% -----------------------------------------------------------------------
signal_table = {
    % --- Test 1: Type A, L36, DOC0.5, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 2: Type A, L36, DOC1, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 3: Type A, L36, DOC2, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 4: Type A, L36, DOC4, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 5: Type A, L36, DOC6, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 6: Type A, L36, DOC0.5, WOC2, N2700, F270 → Stable ---
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 7: Type A, L36, DOC1, WOC2, N2700, F270 → Stable ---
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 8: Type A, L36, DOC1.5, WOC2, N2700, F270 → Stable ---
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 9: Type A, L36, DOC2, WOC2, N2700, F270 → Chatter ---
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 10: Type A, L36, DOC4, WOC2, N2700, F270 → Chatter ---
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 11: Type A, L36, DOC1, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 12: Type A, L36, DOC1.5, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 13: Type A, L36, DOC2, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 14: Type A, L36, DOC3, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 15: Type A, L36, DOC0.5, WOC1, N3600, F360 → Stable ---
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 16: Type A, L36, DOC1, WOC1, N3600, F360 → Stable ---
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 17: Type A, L36, DOC1.5, WOC1, N3600, F360 → Stable ---
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 18: Type A, L36, DOC2, WOC1, N3600, F360 → Chatter ---
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';

    % --- Test 19: Type A, L36, DOC3, WOC1, N3600, F360 → Chatter ---
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';
% ------------------------------------------------------------------------------------------------
    % --- Test 20: Type B, L56, DOC0.5, WOC1, N2700, F270 → Stable ---
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 21: Type B, L56, DOC1, WOC1, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 22: Type B, L56, DOC1.5, WOC1, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 23: Type B, L56, DOC2, WOC1, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 24: Type B, L56, DOC4, WOC1, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 25: Type B, L56, DOC6, WOC1, N2700, F270 → Chatter (as per table: R1,R3,R4,R5) ---
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 26: Type B, L56, DOC0.5, WOC2, N2700, F270 → Stable ---
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 27: Type B, L56, DOC1, WOC2, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 28: Type B, L56, DOC1.5, WOC2, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 29: Type B, L56, DOC2, WOC2, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 30: Type B, L56, DOC4, WOC2, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 31: Type B, L56, DOC0.5, WOC1, N1800, F180 → Stable ---
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 32: Type B, L56, DOC1, WOC1, N1800, F180 → Stable ---
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 33: Type B, L56, DOC1.5, WOC1, N1800, F180 → Stable ---
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 34: Type B, L56, DOC2, WOC1, N1800, F180 → Chatter ---
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R4.wav',  0, 'Chatter';

    % --- Test 35: Type B, L56, DOC3, WOC1, N1800, F180 → Chatter ---
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R4.wav',  0, 'Chatter';

    % --- Test 36: Type B, L56, DOC0.5, WOC1, N3600, F360 → Stable ---
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 37: Type B, L56, DOC1, WOC1, N3600, F360 → Stable ---
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 38: Type B, L56, DOC1.5, WOC1, N3600, F360 → Stable ---
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 39: Type B, L56, DOC2, WOC1, N3600, F360 → Chatter (corrected L36 → L56) ---
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';

    % --- Test 40: Type B, L56, DOC3, WOC1, N3600, F360 → Chatter (corrected L36 → L56) ---
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';
};

% باقي الكود (استخراج الميزات، المعالجة، الحفظ) يبقى كما هو دون تغيير

%% =======================================================================
%  Window Settings
%  - DeltaT = 20 ms  (≈ one spindle revolution at 2700 RPM,
%    dominant speed across experiments — kept fixed)
%  - OverlapRatio = 0.5  (50% overlap, standard practice)
%  - TARGET_WINDOWS = 350  (fits comfortably in ~4 s signals)
%% =======================================================================
DeltaT           = 20e-3;
OverlapRatio     = 0.5;
TARGET_WINDOWS   = 350;

% Feature schema for the corrected extraction used in the new study.
% The explicit list fixes both the number and the order of the 40 features.
FeatureSchemaVersion = 'conservative-v8-2026-09-03';
FeatureNames = { ...
    'Mean', ...                                      %  1
    'Median', ...                                    %  2
    'Standard_Deviation', ...                        %  3
    'Hjorth_Complexity', ...                         %  4
    'Coefficient_of_Variation', ...                  %  5
    'RMS', ...                                       %  6
    'Peak_Absolute_Amplitude', ...                   %  7
    'Peak_to_Peak', ...                              %  8
    'Skewness', ...                                  %  9
    'Kurtosis', ...                                  % 10 (Pearson; Gaussian = 3)
    'Crest_Factor', ...                              % 11
    'Mean_Absolute_Amplitude', ...                   % 12
    'Square_Root_Amplitude', ...                     % 13
    'Clearance_Factor', ...                          % 14
    'Shape_Factor', ...                              % 15
    'Impulse_Factor', ...                            % 16
    'Skewness_Factor', ...                           % 17 (Reference 7 definition)
    'Kurtosis_Factor', ...                           % 18 (Reference 7 definition)
    'Zero_Crossing_Rate', ...                        % 19
    'Lag1_Autocorrelation', ...                      % 20
    'Time_Domain_Energy', ...                        % 21
    'Window_to_Record_Energy_Ratio', ...             % 22
    'Mean_Spectral_Power', ...                       % 23
    'Power_Spectral_Centroid', ...                   % 24
    'RMS_Frequency', ...                             % 25
    'Magnitude_Spectral_Centroid', ...               % 26
    'Magnitude_Spectral_Bandwidth', ...              % 27
    'Frequency_Standard_Deviation', ...              % 28
    'Power_Spectral_Skewness', ...                   % 29
    'Power_Spectral_Kurtosis', ...                   % 30 (Pearson; non-excess)
    'Median_Frequency', ...                          % 31
    'Spectral_Rolloff_85', ...                       % 32
    'Spectral_Energy', ...                           % 33
    'Spectral_Flatness', ...                         % 34
    'Spectral_Entropy', ...                          % 35
    'Peak_Frequency_Ratio', ...                      % 36
    'MPE_s4', ...                                    % 37
    'DC_Removed_Envelope_Spectrum_Crest_Factor', ... % 38
    'Wavelet_Packet_Energy_Entropy', ...             % 39
    'wRCMDE_s1' ...                                  % 40
    };
assert(numel(FeatureNames) == 40, ...
    'The corrected feature schema must contain exactly 40 features.');
%% =======================================================================
%  [v8] Entropy/Envelope
%  Feature Parameters
%  Three feature families are added below, computed per the cited
%  papers' equations and reported parameter choices. The v7 corrections to
%  the other features are documented next to their calculations below.
%
%  (A) wRCMDE — weighted Refined Composite Multiscale Dispersion Entropy
%      Yang, B.; Guo, K.; Sun, J. "Chatter Detection in Robotic Milling
%      Using Entropy Features." Appl. Sci. 2022, 12, 8276.
%      https://doi.org/10.3390/app12168276
%      - DisEn defined by Eqs. (5)-(8); RCMDE by Eqs. (11)-(12);
%        wRCMDE (kurtosis-weighted RCMDE) by Eqs. (13)-(14).
%      - Parameters per paper Section 4.1: m = 5, c = 5, tau = 1.
%      - IN THE SOURCE PAPER, wRCMDE is genuinely a 4-DIMENSIONAL
%        feature (s = 1, 2, 3, 4 used together as separate SVM inputs,
%        per their Figure 5/6) — it is NOT reduced to one scale by Yang
%        et al. To match this project's 40-feature target (one row per
%        nonlinear descriptor), a SINGLE scale, s_wRCMDE (below), is
%        used instead. This is OUR choice, not a literal reproduction
%        of the source paper's feature set, and should be described as
%        such in the manuscript. s_wRCMDE = 1 is used as a reasonable
%        default (the finest, least coarse-grained scale, where
%        differences between classes tend to be largest before
%        coarse-graining smooths them out), but this has not been
%        empirically verified on your data. Before finalizing, consider
%        running the same kind of scale-factor comparison Liu et al.
%        perform for MPE (Section 5.2 of their paper) on your own
%        labeled windows, and report whichever scale gives the cleanest
%        class separation, citing that as your own analysis rather than
%        attributing the specific scale choice to Yang et al.
%
%  (B) MPE — Multi-scale Permutation Entropy
%      Liu, X.; Wang, Z.; Li, M.; Yue, C.; Liang, S.Y.; Wang, L.
%      "Feature extraction of milling chatter based on optimized
%      variational mode decomposition and multi-scale permutation
%      entropy." Int J Adv Manuf Technol 2021.
%      https://doi.org/10.1007/s00170-021-07027-0
%      - Coarse-graining per their Eq. (5) (same scheme as Costa et al.,
%        2002); base entropy is the classical permutation entropy of
%        Bandt & Pompe (2002).
%      - Parameters per paper Section 5.2: m = 6, tau = 1.
%      - Liu et al. identify s = 4 as the most discriminative scale, so
%        s_MPE = 4 is fixed below. Their complete method computes MPE after
%        optimized VMD and signal reconstruction; this script instead applies
%        it directly to raw windows and must not be described as a literal
%        reproduction of the complete Liu et al. pipeline.
%
%  (C) CE — Crest factor of the envelope spectrum
%      Liu, X.; Wang, Z.; Li, M.; Yue, C.; Liang, S.Y.; Wang, L. (2021),
%      Int J Adv Manuf Technol, Eqs. (6)-(9). In the source paper, CE is
%      used as the PSO fitness function for selecting VMD parameters
%      [K, alpha]; here it is extracted directly from the raw window as
%      a stand-alone descriptor based on Eq. (9):
%      CE = max(E)/RMS(E), where E is the FFT of the Hilbert-transform
%      envelope of the signal (Eqs. 6-8). No extra parameters (m, c, tau)
%      are required for CE.
%      IMPLEMENTATION NOTE: the envelope signal is non-negative by
%      construction, so its spectrum has a very large 0 Hz (DC) term
%      that is not related to periodic impulse content. Consistent with
%      standard envelope-spectrum/demodulation practice (the resonance-
%      demodulation literature underlying Eq. 9, e.g., Zhang et al.
%      2015, cited as Ref. [24] in Liu et al.), the DC component is
%      removed before the spectrum is taken, so CE reflects periodic
%      impulsiveness rather than the envelope's mean level. This is an
%      implementation choice not made explicit in the paper's text and
%      should be reported as such if asked by a reviewer.
%
%  NOTE ON WINDOW LENGTH: all three source-paper computations above use
%  signal segments of ~1 to ~10 s (thousands of samples). Your
%  per-window length here is DeltaT = 20 ms, i.e. only a few hundred
%  samples per window (fewer still after coarse-graining). With m = 5/
%  c = 5 (3125 possible dispersion patterns) or m = 6 (720 possible
%  permutation patterns), a 20 ms window may not contain enough points
%  to populate the pattern space reliably, which can bias the entropy
%  estimate toward lower/noisier values. Consider reporting this as a
%  limitation, or re-checking results with a longer DeltaT (or computing
%  these features once per full signal rather than per window) before
%  relying on them for your final feature set.
%% =======================================================================
m_DE       = 5;     % embedding dimension for dispersion entropy (DisEn/RCMDE)
c_DE       = 5;     % number of classes for dispersion entropy
tau_DE     = 1;     % time delay for dispersion entropy
s_wRCMDE   = 1;     % single chosen scale for wRCMDE — OUR choice (see note
                     % above); not the literal Yang et al. 4-feature set,
                     % and not yet empirically verified on your data

m_PE   = 6;     % embedding dimension for permutation entropy (MPE)
tau_PE = 1;     % time delay for permutation entropy
s_MPE  = 4;     % scale factor selected in Liu et al.'s scale comparison
assert(s_MPE == 4 && s_wRCMDE == 1, ...
    'FeatureNames encode MPE_s4 and wRCMDE_s1; update names if scales change.');
%% =======================================================================
%  Main Processing Loop
%% =======================================================================
num_files = size(signal_table, 1);

for file_idx = 1:num_files

    filename    = signal_table{file_idx, 1};
    label_val   = signal_table{file_idx, 2};   % 0=Chatter, 1=Stable
    label_str   = signal_table{file_idx, 3};

    % -------------------------------------------------------------------
    % Check file existence
    % -------------------------------------------------------------------
    if ~exist(filename, 'file')
        warning('[SKIP] File not found: %s', filename);
        continue;
    end

    fprintf('Processing [%d/%d] (%s): %s\n', ...
        file_idx, num_files, label_str, filename);

    % -------------------------------------------------------------------
    % Read signal
    % -------------------------------------------------------------------
    [signal, fs] = audioread(filename);
    signal = signal(:, 1);          % Use channel 1 if stereo

    WindowSamples = round(DeltaT * fs);   % samples per window (e.g. 320 @ 16kHz)
    StepSamples   = round((1 - OverlapRatio) * WindowSamples);  % 50% overlap step

    % -------------------------------------------------------------------
    % Determine available windows from signal length
    % -------------------------------------------------------------------
    available_windows = floor((length(signal) - WindowSamples) / StepSamples) + 1;

    if available_windows <= 0
        warning('[SKIP] Signal too short to extract even one window: %s', filename);
        continue;
    end

    % -------------------------------------------------------------------
    % Decide number of windows to extract
    %
    %   Case A — signal long enough (normal case ~4 s):
    %            extract exactly TARGET_WINDOWS windows,
    %            starting from index 1.
    %
    %   Case B — signal shorter than needed for TARGET_WINDOWS:
    %            extract as many windows as available
    %            (Option 2 per specification).
    % -------------------------------------------------------------------
    num_windows = min(available_windows, TARGET_WINDOWS);

    % -------------------------------------------------------------------
    % Pre-allocate feature struct array
    % -------------------------------------------------------------------
    SigData = repmat(struct( ...
        'FileName',                                      [], ...
        'Label',                                         [], ...
        'Mean',                                          [], ...
        'Median',                                        [], ...
        'Standard_Deviation',                             [], ...
        'Hjorth_Complexity',                             [], ...
        'Coefficient_of_Variation',                      [], ...
        'RMS',                                           [], ...
        'Peak_Absolute_Amplitude',                        [], ...
        'Peak_to_Peak',                                  [], ...
        'Skewness',                                      [], ...
        'Kurtosis',                                      [], ...
        'Crest_Factor',                                  [], ...
        'Mean_Absolute_Amplitude',                        [], ...
        'Square_Root_Amplitude',                          [], ...
        'Clearance_Factor',                               [], ...
        'Shape_Factor',                                   [], ...
        'Impulse_Factor',                                 [], ...
        'Skewness_Factor',                               [], ...
        'Kurtosis_Factor',                               [], ...
        'Zero_Crossing_Rate',                             [], ...
        'Lag1_Autocorrelation',                           [], ...
        'Time_Domain_Energy',                             [], ...
        'Window_to_Record_Energy_Ratio',                  [], ...
        'Mean_Spectral_Power',                            [], ...
        'Power_Spectral_Centroid',                        [], ...
        'RMS_Frequency',                                  [], ...
        'Magnitude_Spectral_Centroid',                    [], ...
        'Magnitude_Spectral_Bandwidth',                   [], ...
        'Frequency_Standard_Deviation',                   [], ...
        'Power_Spectral_Skewness',                       [], ...
        'Power_Spectral_Kurtosis',                       [], ...
        'Median_Frequency',                               [], ...
        'Spectral_Rolloff_85',                            [], ...
        'Spectral_Energy',                               [], ...
        'Spectral_Flatness',                              [], ...
        'Spectral_Entropy',                              [], ...
        'Peak_Frequency_Ratio',                          [], ...
        'MPE_s4',                                         [], ...
        'DC_Removed_Envelope_Spectrum_Crest_Factor',      [], ...
        'Wavelet_Packet_Energy_Entropy',                  [], ...
        'wRCMDE_s1',                                      []  ...
        ), num_windows, 1);

    stored_fields = fieldnames(SigData);
    assert(isequal(stored_fields(3:end), FeatureNames(:)), ...
        'SigData field order does not match FeatureNames.');
  % 'LabelStr',           [], ...
    % -------------------------------------------------------------------
    % Energy of full original signal (used for EnR)
    % -------------------------------------------------------------------
    OriginalSigEnergy = sum(signal .^ 2);

    % -------------------------------------------------------------------
    % Window loop — exactly num_windows iterations
    % -------------------------------------------------------------------
    [~, name_only, ~] = fileparts(filename);

    for co = 1:num_windows

        % Start sample of this window
        i_start   = (co - 1) * StepSamples + 1;
        i_end     = i_start + WindowSamples - 1;

        % Safety guard (should not trigger for Case A)
        if i_end > length(signal)
            warning('[WINDOW] Window %d exceeds signal length in %s. Stopping early.', ...
                co, filename);
            SigData(co:end) = [];
            break;
        end

        % *** RAW signal window — no noise removal ***
        SigSaving = signal(i_start : i_end);

        %% ---- Frequency Domain Setup (raw, rectangular-window spectrum) ----
        % A_k is the one-sided amplitude spectrum used by Reference 7 and by
        % the manuscript table. S_k = A_k^2 is used only for explicitly
        % power-weighted descriptors. This restores the original convention
        % for descriptors that did not require correction.
        N = length(SigSaving);
        Y = fft(double(SigSaving));
        positive_bins = 1 : (floor(N / 2) + 1);
        magnitude_spectrum = abs(Y(positive_bins)) / N;

        if rem(N, 2) == 0
            bins_to_double = 2 : (numel(positive_bins) - 1); % exclude DC and Nyquist
        else
            bins_to_double = 2 : numel(positive_bins);       % exclude DC only
        end
        magnitude_spectrum(bins_to_double) = 2 * magnitude_spectrum(bins_to_double);
        power_spectrum = magnitude_spectrum .^ 2;
        frequencies = fs * (0 : (numel(positive_bins) - 1))' / N;

        %% ---- Metadata ----
        SigData(co).FileName  = name_only;
        SigData(co).Label     = label_val;    % 0 / 1
        % SigData(co).LabelStr  = label_str;    % 'Chatter' / 'Stable'

        %% ========== Time-Domain Features (1–22) ==========

        % [1]  Mean
        window_mean = mean(SigSaving);
        centered_signal = SigSaving - window_mean;
        central_moment_2 = mean(centered_signal .^ 2);
        central_moment_3 = mean(centered_signal .^ 3);
        central_moment_4 = mean(centered_signal .^ 4);
        SigData(co).Mean = window_mean;
        % [2]  Median
        SigData(co).Median = median(SigSaving);
        % [3]  Standard Deviation
        SigData(co).Standard_Deviation = std(SigSaving, 0); % N-1 denominator
        % [4]  Hjorth complexity (Hjorth, 1970).
        % Replaces Variance because Variance = Standard_Deviation^2.
        first_difference = diff(SigSaving);
        second_difference = diff(first_difference);
        signal_activity = var(SigSaving, 1);
        first_difference_activity = var(first_difference, 1);
        second_difference_activity = var(second_difference, 1);
        hjorth_mobility = sqrt(max(localSafeDivide( ...
            first_difference_activity, signal_activity), 0));
        derivative_mobility = sqrt(max(localSafeDivide( ...
            second_difference_activity, first_difference_activity), 0));
        SigData(co).Hjorth_Complexity = localSafeDivide( ...
            derivative_mobility, hjorth_mobility);
        % [5]  Coefficient of Variation
        SigData(co).Coefficient_of_Variation = localSafeDivide( ...
            SigData(co).Standard_Deviation, window_mean);
        % [6]  RMS
        SigData(co).RMS = sqrt(mean(SigSaving .^ 2));
        % [7]  Peak
        SigData(co).Peak_Absolute_Amplitude = max(abs(SigSaving));
        % [8]  Peak-to-Peak
        SigData(co).Peak_to_Peak = max(SigSaving) - min(SigSaving);
        % [9-10] Standard moment coefficients (Pearson kurtosis).
        % These are the conventional population-moment definitions; no
        % finite-sample bias correction is applied within each window.
        SigData(co).Skewness = localSafeDivide( ...
            central_moment_3, central_moment_2 ^ (3 / 2));
        SigData(co).Kurtosis = localSafeDivide( ...
            central_moment_4, central_moment_2 ^ 2);
        % [11] Crest Factor
        SigData(co).Crest_Factor = localSafeDivide( ...
            SigData(co).Peak_Absolute_Amplitude, SigData(co).RMS);
        % [12] Average Amplitude
        SigData(co).Mean_Absolute_Amplitude = mean(abs(SigSaving));
        % [13] Square Root Amplitude
        SigData(co).Square_Root_Amplitude = (mean(sqrt(abs(SigSaving)))) ^ 2;
        % [14] Clearance Factor
        SigData(co).Clearance_Factor = localSafeDivide( ...
            SigData(co).Peak_Absolute_Amplitude, SigData(co).Square_Root_Amplitude);
        % [15] Shape Factor
        SigData(co).Shape_Factor = localSafeDivide( ...
            SigData(co).RMS, SigData(co).Mean_Absolute_Amplitude);
        % [16] Impulse Factor
        SigData(co).Impulse_Factor = localSafeDivide( ...
            SigData(co).Peak_Absolute_Amplitude, SigData(co).Mean_Absolute_Amplitude);
        % [17-18] Skewness and kurtosis factor forms from Reference 7.
        % These are scale-dependent ratios; they are not dimensionless.
        SigData(co).Skewness_Factor = localSafeDivide( ...
            SigData(co).Skewness, SigData(co).RMS ^ 3);
        SigData(co).Kurtosis_Factor = localSafeDivide( ...
            SigData(co).Kurtosis, SigData(co).RMS ^ 4);
        % [19] Zero Crossing Rate (Reference-7 Table-2 convention)
        SigData(co).Zero_Crossing_Rate = ...
            sum(abs(diff(sign(SigSaving)))) / (2 * N);

        % [20] One-Step Auto-correlation Function (OSAF)
        signal_sum = sum(SigSaving);
        signal_square_sum = sum(SigSaving .^ 2);
        adjacent_product_sum = sum(SigSaving(1:end-1) .* SigSaving(2:end));
        lag1_numerator = N * adjacent_product_sum - signal_sum ^ 2;
        lag1_denominator = N * signal_square_sum - signal_sum ^ 2;
        SigData(co).Lag1_Autocorrelation = localSafeDivide( ...
            lag1_numerator, lag1_denominator);

        % [21] Time Domain Energy
        SigData(co).Time_Domain_Energy = sum(SigSaving .^ 2);
        % [22] Energy Ratio
        SigData(co).Window_to_Record_Energy_Ratio = localSafeDivide( ...
            SigData(co).Time_Domain_Energy, OriginalSigEnergy);

        %% ========== Frequency-Domain Features (23–36) ==========

        total_spectral_power = sum(power_spectrum);
        total_spectral_magnitude = sum(magnitude_spectrum);

        % [23] Mean spectral power per positive-frequency bin.
        % The old name "Mean of Frequency" was dimensionally incorrect.
        SigData(co).Mean_Spectral_Power = mean(power_spectrum);
        % [33] Spectral energy under the manuscript convention S_k = A_k^2.
        SigData(co).Spectral_Energy = total_spectral_power;

        if total_spectral_power > 0
            % [24] Power spectral centroid (centre frequency), Hz
            power_centroid = sum(frequencies .* power_spectrum) / total_spectral_power;
            SigData(co).Power_Spectral_Centroid = power_centroid;

            % The second raw moment is retained only as an intermediate.
            mean_square_frequency = ...
                sum((frequencies .^ 2) .* power_spectrum) / total_spectral_power;

            % [25] RMS frequency: square root of second raw moment, Hz
            SigData(co).RMS_Frequency = sqrt(max(mean_square_frequency, 0));

            % The frequency variance is retained only as an intermediate.
            frequency_variance = sum(((frequencies - power_centroid) .^ 2) .* ...
                power_spectrum) / total_spectral_power;

            % [28] Frequency standard deviation, Hz
            frequency_standard_deviation = sqrt(max(frequency_variance, 0));
            SigData(co).Frequency_Standard_Deviation = frequency_standard_deviation;

            % [29-30] Standardized central spectral moments (power weighted).
            % These replace MSF and VF, which were exact squares of features
            % 25 and 28. Definitions follow the scaled spectral descriptors
            % documented by MathWorks (after Peeters, 2004).
            third_spectral_moment = sum(((frequencies - power_centroid) .^ 3) .* ...
                power_spectrum);
            fourth_spectral_moment = sum(((frequencies - power_centroid) .^ 4) .* ...
                power_spectrum);
            SigData(co).Power_Spectral_Skewness = localSafeDivide( ...
                third_spectral_moment, frequency_standard_deviation ^ 3 * ...
                total_spectral_power);
            SigData(co).Power_Spectral_Kurtosis = localSafeDivide( ...
                fourth_spectral_moment, frequency_standard_deviation ^ 4 * ...
                total_spectral_power);
        else
            SigData(co).Power_Spectral_Centroid = NaN;
            SigData(co).RMS_Frequency = NaN;
            SigData(co).Frequency_Standard_Deviation = NaN;
            SigData(co).Power_Spectral_Skewness = NaN;
            SigData(co).Power_Spectral_Kurtosis = NaN;
        end

        % [26-27,31-32,34-36] Reference-7 magnitude-spectrum convention.
        if total_spectral_magnitude > 0
            magnitude_centroid = sum(frequencies .* magnitude_spectrum) / ...
                total_spectral_magnitude;
            SigData(co).Magnitude_Spectral_Centroid = magnitude_centroid;
            SigData(co).Magnitude_Spectral_Bandwidth = sqrt(sum( ...
                ((frequencies - magnitude_centroid) .^ 2) .* magnitude_spectrum) / ...
                total_spectral_magnitude);

            cumulative_magnitude = cumsum(magnitude_spectrum);
            [~, median_idx] = min(abs(cumulative_magnitude - ...
                0.50 * total_spectral_magnitude));
            SigData(co).Median_Frequency = frequencies(median_idx);

            rolloff_idx = find(cumulative_magnitude >= ...
                0.85 * total_spectral_magnitude, 1, 'first');
            SigData(co).Spectral_Rolloff_85 = frequencies(rolloff_idx);

            if any(magnitude_spectrum == 0)
                SigData(co).Spectral_Flatness = 0;
            else
                SigData(co).Spectral_Flatness = ...
                    exp(mean(log(magnitude_spectrum))) / mean(magnitude_spectrum);
            end

            spectral_probability = magnitude_spectrum / total_spectral_magnitude;
            positive_probability = spectral_probability(spectral_probability > 0);
            SigData(co).Spectral_Entropy = ...
                -sum(positive_probability .* log2(positive_probability));
            SigData(co).Peak_Frequency_Ratio = ...
                max(magnitude_spectrum) / total_spectral_magnitude;
        else
            SigData(co).Magnitude_Spectral_Centroid = NaN;
            SigData(co).Magnitude_Spectral_Bandwidth = NaN;
            SigData(co).Median_Frequency = NaN;
            SigData(co).Spectral_Rolloff_85 = NaN;
            SigData(co).Spectral_Flatness = NaN;
            SigData(co).Spectral_Entropy = NaN;
            SigData(co).Peak_Frequency_Ratio = NaN;
        end

        %% ========== Features 37-40 ==========
        % MPE, envelope-spectrum crest factor, WPEE, and single-scale
        % wRCMDE.

        % [39] Wavelet Packet Energy Entropy (WPEE)
        wavelet_name = 'db4';
        level        = 5;
        tree = wpdec(SigSaving, level, wavelet_name);
        nodes = leaves(tree);
        energy_wp = zeros(length(nodes), 1);
        for k = 1:length(nodes)
            coeff = wpcoef(tree, nodes(k));
            energy_wp(k) = sum(coeff .^ 2);
        end
        total_energy_wp = sum(energy_wp);
        if total_energy_wp > 0
            prob_wp = energy_wp / total_energy_wp;
            positive_prob_wp = prob_wp(prob_wp > 0);
            SigData(co).Wavelet_Packet_Energy_Entropy = ...
                -sum(positive_prob_wp .* log2(positive_prob_wp));
        else
            SigData(co).Wavelet_Packet_Energy_Entropy = NaN;
        end

        % See the parameter block earlier in this script for why wRCMDE is a
        % single chosen scale here (our choice, flagged).

        % [40] wRCMDE at scale factor s = s_wRCMDE (our chosen single
        %      scale — see parameter block note; NOT the literal 4-scale
        %      feature set Yang et al. use in their own SVM).
        %   Yang, Guo & Sun (2022), Appl. Sci. 12, 8276, Eqs. (5)-(14).
        SigData(co).wRCMDE_s1 = ...
            localWRCMDE(SigSaving, m_DE, c_DE, tau_DE, s_wRCMDE);

        % [37] MPE at scale factor s = s_MPE (= 4, selected by Liu et al.).
        %   Liu, Wang, Li, Yue, Liang & Wang (2021), Int J Adv Manuf
        %   Technol, coarse-graining Eq. (5) + Bandt-Pompe (2002)
        %   permutation entropy.
        SigData(co).MPE_s4 = localMPE(SigSaving, m_PE, tau_PE, s_MPE);

        % [38] Crest factor of the DC-removed envelope spectrum
        %   Liu, Wang, Li, Yue, Liang & Wang (2021), Eqs. (6)-(9).
        SigData(co).DC_Removed_Envelope_Spectrum_Crest_Factor = ...
            localEnvelopeSpectrumCE(SigSaving);
    end % window loop

    % -------------------------------------------------------------------
    % Save — one .mat file per signal, same base name
    % -------------------------------------------------------------------
    save_name = [name_only '.mat'];
    save(save_name, 'SigData', 'FeatureNames', 'FeatureSchemaVersion');
    fprintf('  Saved: %s (%d windows, label=%d [%s])\n', ...
        save_name, length(SigData), label_val, label_str);

end % file loop

fprintf('\nAll files processed.\n');


%% =======================================================================
%  [NEW in v4] Local Functions — Entropy Feature Implementations
%  Defined after script code (valid in MATLAB R2016b+ script files).
%  Each function implements the cited paper's equations exactly.
%% =======================================================================

function out = localDispersionEntropy(u, m, c, tau)
% Dispersion Entropy (DisEn).
% Yang, Guo & Sun (2022), Appl. Sci. 12, 8276, Eqs. (5)-(8).
    u = double(u(:));
    L = length(u);
    mu = mean(u);
    sg = std(u);
    if sg == 0
        out = 0;
        return;
    end

    y = normcdf(u, mu, sg);          % Eq. (5): normal-CDF mapping to [0,1]
    z = round(c .* y + 0.5);
    z(z < 1) = 1;
    z(z > c) = c;

    Nvec = L - (m - 1) * tau;
    if Nvec < 1
        out = NaN;
        return;
    end

    patternIdx = zeros(Nvec, 1);
    for i = 1:Nvec
        emb = z(i : tau : i + (m - 1) * tau);   % Eq. (6): embedding vector
        idx = 0;
        for d = 1:m
            idx = idx * c + (emb(d) - 1);        % unique base-c pattern id
        end
        patternIdx(i) = idx;                      % range: 0 .. c^m - 1
    end

    edges  = -0.5 : 1 : (c^m - 0.5);
    counts = histcounts(patternIdx, edges);
    p      = counts(counts > 0) / Nvec;           % Eq. (7)
    out    = -sum(p .* log(p));                   % Eq. (8)
end

function y = localCoarseGrainOffset(u, s, k)
% Refined-composite coarse-graining with starting offset k.
% Yang, Guo & Sun (2022), Eq. (11).
    u = double(u(:));
    L = length(u);
    starts = k : s : (L - s + 1);
    nWin = numel(starts);
    y = zeros(nWin, 1);
    for j = 1:nWin
        y(j) = mean(u(starts(j) : starts(j) + s - 1));
    end
end

function out = localRCMDE(u, m, c, tau, s)
% Refined Composite Multiscale Dispersion Entropy (RCMDE).
% Yang, Guo & Sun (2022), Eq. (12).
    deVals = zeros(s, 1);
    for k = 1:s
        yk = localCoarseGrainOffset(u, s, k);
        deVals(k) = localDispersionEntropy(yk, m, c, tau);
    end
    out = mean(deVals);
end

function out = localWRCMDE(u, m, c, tau, s)
% Weighted Refined Composite Multiscale Dispersion Entropy (wRCMDE).
% Yang, Guo & Sun (2022), Eqs. (13)-(14).
    u = double(u(:));
    centered = u - mean(u);
    moment2 = mean(centered .^ 2);
    if moment2 == 0
        out = NaN;
        return;
    end
    % Eq. (13) is the population Pearson moment coefficient. MATLAB's
    % one-argument kurtosis applies finite-sample bias correction, so the
    % moments are evaluated explicitly here to reproduce the equation.
    KT = mean(centered .^ 4) / (moment2 ^ 2);
    out = KT * localRCMDE(u, m, c, tau, s);
end

function y = localCoarseGrainSimple(x, s)
% Non-overlapping-average coarse-graining used for MPE.
% Liu et al. (2021), Eq. (5) (same scheme as Costa et al., 2002).
    x = double(x(:));
    n = length(x);
    nWin = floor(n / s);
    y = zeros(nWin, 1);
    for j = 1:nWin
        y(j) = mean(x((j - 1) * s + 1 : j * s));
    end
end

function idx = localPermRank(v)
% Lehmer-code-based unique index (1..m!) identifying an ordinal pattern.
    m = length(v);
    [~, ord] = sort(v);
    idx = 0;
    avail = ord;
    for i = 1:m
        rank = sum(avail(i + 1:end) < avail(i));
        idx = idx + rank * factorial(m - i);
    end
    idx = idx + 1;
end

function out = localPermEntropy(x, m, tau)
% Permutation Entropy (Bandt & Pompe, 2002), as used by Liu et al. (2021).
    x = double(x(:));
    L = length(x);
    N = L - (m - 1) * tau;
    if N < 1
        out = NaN;
        return;
    end
    pIdx = zeros(N, 1);
    for i = 1:N
        seg = x(i : tau : i + (m - 1) * tau);
        pIdx(i) = localPermRank(seg);
    end
    edges  = 0.5 : 1 : (factorial(m) + 0.5);
    counts = histcounts(pIdx, edges);
    p      = counts(counts > 0) / N;
    out    = -sum(p .* log(p));
end

function out = localMPE(x, m, tau, s)
% Multi-scale Permutation Entropy (MPE).
% Liu, Wang, Li, Yue, Liang & Wang (2021), Int J Adv Manuf Technol.
    xc  = localCoarseGrainSimple(x, s);
    out = localPermEntropy(xc, m, tau);
end

function out = localEnvelopeSpectrumCE(x)
% [NEW in v5] Crest factor of the envelope spectrum (CE).
% Liu, Wang, Li, Yue, Liang & Wang (2021), Int J Adv Manuf Technol,
% Eqs. (6)-(9):
%   xh(t)  = Hilbert transform of x(t)                       Eq. (6)
%   y(t)   = x(t) + j*xh(t)         (analytic signal)         Eq. (7)
%   A(t)   = sqrt(x(t)^2 + xh(t)^2) (envelope signal)         Eq. (8)
%   E      = FFT(A)                 (envelope spectrum)
%   CE     = max(E) / sqrt(mean(E.^2))                        Eq. (9)
%
% NOTE: the envelope A(t) is non-negative by construction, so its
% spectrum has a very large 0 Hz (DC) term unrelated to periodic
% impulse content; if left in, CE would be approximately constant
% (~sqrt(N)) for any signal and would not discriminate machining
% states. Consistent with standard envelope-spectrum/demodulation
% practice (the resonance-demodulation literature underlying Eq. 9,
% e.g., Zhang et al. 2015, cited as Ref. [24] in Liu et al.), the DC
% component is removed from the envelope before its spectrum is taken.
% This implementation choice is not made explicit in the paper's text
% and should be disclosed if asked.
    x = double(x(:));
    N = length(x);

    % --- Analytic signal via FFT-based Hilbert transform (Eqs. 6-7) ---
    Xf = fft(x);
    H  = zeros(N, 1);
    if mod(N, 2) == 0
        H(1)         = 1;
        H(2 : N/2)   = 2;
        H(N/2 + 1)   = 1;
    else
        H(1)               = 1;
        H(2 : (N+1)/2)     = 2;
    end
    xa = ifft(Xf .* H);

    % --- Envelope signal (Eq. 8), DC removed (see note above) ---
    A = abs(xa);
    A = A - mean(A);

    % --- Envelope spectrum (single-sided), DC bin dropped ---
    Ef    = fft(A);
    Ehalf = abs(Ef(1 : floor(N/2) + 1));
    if numel(Ehalf) > 1
        Ehalf = Ehalf(2:end);
    end

    % --- Crest factor of the envelope spectrum (Eq. 9) ---
    out = localSafeDivide(max(Ehalf), sqrt(mean(Ehalf .^ 2)));
end

function out = localSafeDivide(numerator, denominator)
% Scalar division without the numerical and dimensional bias introduced
% by adding eps to a physical denominator. Undefined ratios are NaN.
    if isfinite(numerator) && isfinite(denominator) && denominator ~= 0
        out = numerator / denominator;
    else
        out = NaN;
    end
end
