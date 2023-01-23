#!/usr/bin/python
import os
import sys
import time


def pv_name(name):
    for pv in globals():
        if name[:2].lower() in pv.lower():
            print("PV:", pv[:])


def pv_return():
    return globals()


# S3
s3_x = "BL23I-AL-SLITS-03:X"
s3_y = "BL23I-AL-SLITS-03:Y"
s3_xcentre = "BL23I-AL-SLITS-03:X:CENTRE"
s3_xcentre_dmov = "BL23I-AL-SLITS-03:X:CENTRE.DMOV"
s3_ycentre = "BL23I-AL-SLITS-03:Y:CENTRE"
s3_ycentre_dmov = "BL23I-AL-SLITS-03:Y:CENTRE.DMOV"
s3_xsize = "BL23I-AL-SLITS-03:X:SIZE"
s3_xsize_dmov = "BL23I-AL-SLITS-03:X:SIZE.DMOV"
s3_ysize = "BL23I-AL-SLITS-03:Y:SIZE"
s3_ysize_dmov = "BL23I-AL-SLITS-03:Y:SIZE.DMOV"

# D2
d2_y = "BL23I-DI-PHDGN-02:FILTER:1"
d2_y_inpos = "BL23I-DI-PHDGN-02:F1:MP:INPOS"
d2_cur_1 = "BL23I-DI-PHDGN-02:OPT1:DIODE1:I:ADC1_VALEGU"
d2_cur_2 = "BL23I-DI-PHDGN-02:OPT1:DIODE2:I:ADC2_VALEGU"
d2_cur_3 = "BL23I-DI-PHDGN-02:OPT1:DIODE3:I:ADC1_VALEGU"
d2_position = {
    "empty": "-145.00",
    "diode 1": "-53.30",
    "diode 2": "-27.90",
    "diode 3": "-2.50",
}
d2_diode1_gain = "BL23I-DI-PHDGN-02:OPT1:DIODE1:GAIN"

# pilatus
pilatus_image_mode = "BL23I-EA-PILAT-01:cam1:ImageMode"
pilatus_trigger_mode = "BL23I-EA-PILAT-01:cam1:TriggerMode"
pilatus_acquire_time = "BL23I-EA-PILAT-01:cam1:AcquireTime"
pilatus_acquire = "BL23I-EA-PILAT-01:cam1:Acquire"
pilatus_acquire_period = "BL23I-EA-PILAT-01:cam1:AcquirePeriod"
pilatus_num_exposures = "BL23I-EA-PILAT-01:cam1:NumExposures"
pilatus_num_images = "BL23I-EA-PILAT-01:cam1:NumImages"
pilatus_file_path = "BL23I-EA-PILAT-01:cam1:FilePath"
pilatus_file_name = "BL23I-EA-PILAT-01:cam1:FileName"
pilatus_filenumber = "BL23I-EA-PILAT-01:cam1:FileNumber"
pilatus_filenumber = "BL23I-EA-PILAT-01:cam1:AutoIncrement"
pilatus_wavelength = "BL24I-EA-PILAT-01:cam1:Wavelength"
pilatus_file_number = "BL23I-EA-PILAT-01:cam1:FileNumber"
pilatus_auto_increment = "BL23I-EA-PILAT-01:cam1:AutoIncrement"
pilatus_file_template = "BL23I-EA-PILAT-01:cam1:FileTemplate"
pilatus_armed = "BL23I-EA-PILAT-01:cam1:Armed"
pilatus_gain = "BL23I-EA-PILAT-01:cam1:Gain"
pilatus_aquire = "BL23I-EA-PILAT-01:cam1:Acquire"
pilatus_detector_state = "BL23I-EA-PILAT-01:cam1:DetectorState_RBV"
pilatus_z_value = "BL23I-EA-DET-01:Z"
pilatus_z_positions = {"out": "239.0", "in": "0.0", "saxs": "50.0"}

# zebra
zebra_pc_arm_sel = "BL23I-EA-ZEBRA-01:ZEBRA:PC_ARM_SEL"
zebra_pc_gate_sel = "BL23I-EA-ZEBRA-01:ZEBRA:PC_GATE_SEL"
zebra_pc_pulse_sel = "BL23I-EA-ZEBRA-01:ZEBRA:PC_PULSE_SEL"
zebra_pc_arm = "BL23I-EA-ZEBRA-01:ZEBRA:PC_ARM"
zebra_pc_disarm = "BL23I-EA-ZEBRA-01:ZEBRA:PC_DISARM"
zebra_pc_gate_inp = "BL23I-EA-ZEBRA-01:ZEBRA:PC_GATE_INP"
zebra_pc_gate_start = "BL23I-EA-ZEBRA-01:ZEBRA:PC_GATE_START"
zebra_pc_gate_wid = "BL23I-EA-ZEBRA-01:ZEBRA:PC_GATE_WID"
zebra_pc_arm_out = "BL23I-EA-ZEBRA-01:ZEBRA:PC_ARM_OUT"

# shutter
shutter_ctrl1 = "BL23I-EA-SHTR-01:CTRL1"
shutter_ctrl2 = "BL23I-EA-SHTR-01:CTRL2"
shutter_status = "BL23I-EA-SHTR-01:STA"

# DCM
dcm_energy = "BL23I-OP-DCM-01:ENERGY"
dcm_energy_dmov = "BL23I-OP-DCM-01:ENERGY.DMOV"

# ID
id_blgapmtr = "SR23I-MO-SERVC-01:BLGAPMTR.VAL"
id_currgapd = "SR23I-MO-SERVC-01:CURRGAPD"
id_allmove = "SR23I-MO-SERVC-01:ALLMOVE"

# SENV
senv_mp_select = "BL23I-EA-SENV-01:MP:SELECT"
senv_mp_inpos = "BL23I-EA-SENV-01:MP:INPOS"
senv_d5_ADC2valegu = "BL23I-DI-PHDGN-05:OPT1:DIODE:I:ADC2_VALEGU"
senv_blx = "BL23I-EA-SENV-01:BLX"
senv_bly = "BL23I-EA-SENV-01:BLY"
senv_blx_rbv = "BL23I-EA-SENV-01:BLX.RBV"
senv_bly_rbv = "BL23I-EA-SENV-01:BLY.RBV"
senv_bluelight_status = "BL23I-EA-ERIO-01:M9:OUT1:TOGGLE"
senv_backlight_status = "BL23I-EA-ERIO-01:M9:OUT2:TOGGLE"

# OAV
oav_cam_mp = "BL23I-DI-OAV-01:CAM:MP:SELECT"

# sample changer
grip_x_rbv = "BL23I-MO-GRIP-01:X.RBV"
grip_x_mv = "BL23I-MO-GRIP-01:X.DMOV"
grip_y_rbv = "BL23I-MO-GRIP-01:Y.RBV"
grip_y_mv = "BL23I-MO-GRIP-01:Y.DMOV"
grip_z_rbv = "BL23I-MO-GRIP-01:Z.RBV"
grip_z_mv = "BL23I-MO-GRIP-01:Z.DMOV"