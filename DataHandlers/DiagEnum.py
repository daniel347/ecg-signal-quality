from enum import Enum


class DiagEnum(Enum):
    AF = 1
    CannotExcludePathology = 2
    NoAF = 3
    PoorQuality = 4
    ScreeningFailure = 5
    Undecided = 6
    ReviewersDisagree = -1

    # Extra labels for feas1
    HeartBlock = 8  # 2nd or 3rd degree
    VentricularTachycardia = 7

    # Extra labels for trial
    OtherSignificantArrythmia = 9


def trialDiagToEnum(diagNum):
    if diagNum == 5:
        return DiagEnum.PoorQuality
    elif diagNum == 4:
        return DiagEnum.OtherSignificantArrythmia
    else:
        return DiagEnum(diagNum)


def feas1DiagToEnum(diagNum):
    if diagNum == 1:
        return DiagEnum.HeartBlock
    elif diagNum == 2:
        return DiagEnum.AF
    elif diagNum == 3:
        return DiagEnum.CannotExcludePathology
    elif diagNum == 4:
        return DiagEnum.NoAF
    else:
        return DiagEnum(diagNum)

