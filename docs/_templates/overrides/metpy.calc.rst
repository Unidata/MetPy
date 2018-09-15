calc
==========

.. automodule:: metpy.calc


   Dry Thermodynamics
   ------------------

   .. autosummary::
      :toctree: ./

      add_height_to_pressure
      add_pressure_to_height
      density
      dry_lapse
      dry_static_energy
      geopotential_to_height
      height_to_geopotential
      height_to_pressure_std
      mean_pressure_weighted
      potential_temperature
      pressure_to_height_std
      sigma_to_pressure
      static_stability
      temperature_from_potential_temperature
      thickness_hydrostatic


   Moist Thermodynamics
   --------------------

   .. autosummary::
      :toctree: ./

      dewpoint
      dewpoint_from_specific_humidity
      dewpoint_rh
      equivalent_potential_temperature
      mixing_ratio
      mixing_ratio_from_relative_humidity
      mixing_ratio_from_specific_humidity
      moist_lapse
      moist_static_energy
      precipitable_water
      psychrometric_vapor_pressure_wet
      relative_humidity_from_dewpoint
      relative_humidity_from_mixing_ratio
      relative_humidity_from_specific_humidity
      relative_humidity_wet_psychrometric
      saturation_equivalent_potential_temperature
      saturation_mixing_ratio
      saturation_vapor_pressure
      specific_humidity_from_mixing_ratio
      thickness_hydrostatic_from_relative_humidity
      vapor_pressure
      vertical_velocity
      vertical_velocity_pressure
      virtual_potential_temperature
      virtual_temperature
      wet_bulb_temperature


   Soundings
   ---------

   .. autosummary::
      :toctree: ./

      bulk_shear
      bunkers_storm_motion
      cape_cin
      critical_angle
      el
      lcl
      lfc
      mixed_layer
      mixed_parcel
      most_unstable_cape_cin
      most_unstable_parcel
      parcel_profile
      parcel_profile_with_lcl
      significant_tornado
      storm_relative_helicity
      supercell_composite
      surface_based_cape_cin


   Dynamic/Kinematic
   -----------------

   .. autosummary::
      :toctree: ./

      absolute_momentum
      absolute_vorticity
      advection
      ageostrophic_wind
      coriolis_parameter
      divergence
      exner_function
      frontogenesis
      geostrophic_wind
      inertial_advective_wind
      kinematic_flux
      montgomery_streamfunction
      potential_vorticity_baroclinic
      potential_vorticity_barotropic
      q_vector
      shearing_deformation
      stretching_deformation
      total_deformation
      vorticity
      wind_components
      wind_direction
      wind_speed


   Boundary Layer/Turbulence
   -------------------------

   .. autosummary::
      :toctree: ./

      brunt_vaisala_frequency
      brunt_vaisala_frequency_squared
      brunt_vaisala_period
      friction_velocity
      tke


   Mathematical Functions
   ----------------------

   .. autosummary::
      :toctree: ./

      cross_section_components
      first_derivative
      gradient
      grid_deltas_from_dataarray
      laplacian
      lat_lon_grid_deltas
      normal_component
      second_derivative
      tangential_component
      unit_vectors_from_cross_section


   Apparent Temperature
   --------------------

   .. autosummary::
      :toctree: ./

      apparent_temperature
      heat_index
      windchill

   Other
   -----

   .. autosummary::
      :toctree: ./

      find_bounding_indices
      find_intersections
      get_layer
      get_layer_heights
      get_perturbation
      isentropic_interpolation
      nearest_intersection_idx
      parse_angle
      reduce_point_density
      resample_nn_1d
      smooth_gaussian
      smooth_n_point


   Deprecated
   ----------

   Do not use these functions in new code, please see their documentation for their replacements.

   .. autosummary::
      :toctree: ./

      get_wind_components
      get_wind_dir
      get_wind_speed
      interp
      interpolate_nans
      lat_lon_grid_spacing
      log_interp
