import os
from textwrap import dedent
import ipywidgets as widgets

from . import util


def get_custom_config_lines(custom_config_path):
    with open(custom_config_path, 'r') as f:
        custom_lines = f.readlines()
    
    return [
        l for l in custom_lines
        if l.strip() and not l.strip().startswith("#")
    ]

def common_option_buttons():
    cpu_count = os.cpu_count()
    multithread_option = util.select_parameter(["Do not use multithreaded processing",
                                         f"Use my {cpu_count} available cores for multithreaded processing"],
                                             description="Select a multithreaded processing option:")
    
    min_coherence_option = util.select_parameter(["Use MintPy's default 0.85 minimum spatial coherence threshold when selecting a reference point",
                                          "Set a minimum spatial coherence threshold for selecting a reference point",
                                         ],
                                          description="Select a spatial coherence threshold option:")
    
    ref_point_description = "Select a reference point option (random pixel above minimum spatial coherence threshold):"
    ref_point_option = util.select_parameter(["Allow MintPy to determine a reference point", 
                                             "Define a reference point"],
                                           description=ref_point_description)

    ref_date_option = util.select_parameter(["Reference time-series to earliest date in stack",
                                            "Allow MintPy to determine reference date"],
                                          description="Select a reference date option:")

    tropo_correct_option = util.select_parameter(["Do not perform tropospheric correction",
                                            "Perform tropospheric correction"],
                                          description="Select a tropospheric correction option:")
    
    deramp_option = util.select_parameter(["Do not perform deramping",
                                          "Deramp method: linear",
                                          "Deramp method: quadratic"
                                         ],
                                          description="Select a phase deramping option:")
    
    unwrapp_err_description = "Select an unwrapping error correction option (not valid for INSAR_GAMMA HyP3 data):"
    unwrapping_error_option = util.select_parameter(["Do not perform unwrapping error correction",
                                          "Unwrapping error method: bridging",
                                          "Unwrapping error method: phase_closure",
                                          "Unwrapping error method: bridging+phase_closure",
                                         ],
                                          description=unwrapp_err_description)

    return {
        "multithread": multithread_option,
        "min_coherence": min_coherence_option,
        "ref_point": ref_point_option,
        "ref_date": ref_date_option,
        "tropo_correct": tropo_correct_option,
        "deramp": deramp_option,
        "unwrapping_error": unwrapping_error_option,
    }


def create_updated_common_option_config(common_option_dict, custom_lines):
    updated_config = []
    multithread = "Use" in common_option_dict["multithread"].value
    mintpy_ref_point = "Allow" in common_option_dict["ref_point"].value
    mintpy_ref_date = "Allow" in common_option_dict["ref_date"].value
    tropo_correct = "Do not" not in common_option_dict["tropo_correct"].value
    deramp = "Do not" not in common_option_dict["deramp"].value
    bridging = "bridging" in common_option_dict["unwrapping_error"].value
    phase_closure = "phase_closure" in common_option_dict["unwrapping_error"].value
    no_unwrap_correct = "Do not" in common_option_dict["unwrapping_error"].value
    min_coherence = "MintPy's default" not in common_option_dict["min_coherence"].value

    # Add all existing custom config options except those that might be manually set by user
    manually_set_options = [
            "compute", 
            "reference",
            "troposphericDelay",
            "unwrapError",
            "deramp",
            "minCoherence"
        ]
    
    for l in custom_lines:
        if all(x not in l for x in manually_set_options):
            updated_config.append(l)

    ### Multithreading ###
    if multithread:
        updated_config.append("mintpy.compute.cluster = local")
        cpu_count = os.cpu_count()
        updated_config.append(f"mintpy.compute.numWorker = {cpu_count}")

    ### Minimum spatial coherence threshold ###
    if min_coherence:
        print("-" * 80)
        min_coherence_value = input("Enter a reference point minimum spatial coherence value (0-1, default: 0.85)")
        updated_config.append(f"mintpy.reference.minCoherence = {min_coherence_value}")
        
    ### Reference date ###
    if not mintpy_ref_date:
        updated_config.append(f"mintpy.reference.date = no")
    else:
        updated_config.append(f"mintpy.reference.date = auto")

    ### Tropospheric correction ###
    if tropo_correct:
        print("-" * 80)
        tropo_method = input('Enter a troposheric delay correction method ("pyaps" for S1, "opera" for NISAR):')
        updated_config.append(f"mintpy.troposphericDelay.method = {tropo_method}")
    else:
        updated_config.append("mintpy.troposphericDelay.method = no")

    ### Reference point ###
    if not mintpy_ref_point:
        is_float = False
        while not is_float:
            try:
                print("-" * 80)
                lat = float(input("Enter reference latitude"))
                lon = float(input("Enter reference longitude"))
                is_float = True
            except ValueError:
                print("Latitude and Longitude must be convertable to float")
                continue
            updated_config.append(f"mintpy.reference.lalo = {lat},{lon}")

    ### Unwrapping error correction ###
    if bridging or phase_closure:
        print("-" * 80)
        print("Discard connected regions smaller than the min area size in pixels.\n")
        conn_comp_min_area = input("Enter the connected component minimum area (default: 2.5e3)")
        updated_config.append(f"mintpy.unwrapError.connCompMinArea = {conn_comp_min_area}")
        method = None
    else:
        method = "no"
        
    if bridging:
        print("-" * 80)
        print(dedent(
            """
            A phase ramp could be estimated based on the largest reliable region, removed from the entire interferogram
            before estimating the phase difference between reliable regions and added back after the correction.
            ("Linear is recommended for L-band data")
            """
        ).lstrip())
        ramp = input("Enter an 'unwrapError.ramp' option (linear, quadratic, no)")
        updated_config.append(f'mintpy.unwrapError.ramp = {ramp}')

        print("-" * 80)
        print("Define the half size of the window used to calculate the median value of phase difference.\n")
        bridge_pts_radius = input("Enter an 'unwrapError.bridgePtsRadius' option (1-inf, default: 50)")
        updated_config.append(f"mintpy.unwrapError.bridgePtsRadius = {bridge_pts_radius}")

        method = "bridging"
        
    if phase_closure:
        print("-" * 80)
        print(dedent(
            """
            Define a region-based strategy to speedup L1-norm regularized least squares inversion.
            Instead of inverting every pixel for the integer ambiguity, a common connected component 
            mask is generated, for each common conn. comp., numSample pixels are radomly selected for 
            inversion, and the median value of the results are used for all pixels within this common connected comp.       
            """
        ).lstrip())
        num_sample = input("Enter an 'unwrapError.numSample' (integer>1, default: 100)")
        updated_config.append(f"mintpy.unwrapError.numSample = {num_sample}")

        method = "phase_closure"

    if bridging and phase_closure:
        method = "bridging+phase_closure"

    if method:
        updated_config.append(f"mintpy.unwrapError.method = {method}")

    ### Deramping ###
    if deramp and "linear" in common_option_dict["deramp"].value:
        deramp_method = "linear"
    elif deramp and "quadratic" in common_option_dict["deramp"].value:
        deramp_method = "quadratic"
    else:
        deramp_method = None
    if deramp_method:
        updated_config.append(f"mintpy.deramp = {deramp_method}")

    return "\n".join(l.rstrip("\n") for l in updated_config)


def interactive_config(custom_lines, full_config_lines):
    param_dict = {}
    
    layout = widgets.Layout(width='initial', height='40px') #set width and height

    custom_config_dict = {}
    for l in custom_lines:
        param = l.split(' ')[0]
        info = l.split('= ')[-1].strip()
        custom_config_dict[param] = info
    
    for i, l in enumerate(full_config_lines):
        if l.startswith('#'):
            param_dict[i] = l
        else:
            param = l.split(' ')[0]
            if param in custom_config_dict.keys():
                info = custom_config_dict[param]
            else:
                info = l.split('= auto ')[-1].strip()[1:]  
            param_dict[param] =  widgets.Text(
                placeholder=info,
                description=f'{param}:',
                disabled=False,
                align_items='stretch', 
                layout = layout,
                style= {'description_width': 'initial'},
            )
    return param_dict, custom_config_dict
  