Calculations ``(metpy.calc)``
=============================

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
      mean_pressure_weighted
      potential_temperature
      sigma_to_pressure
      static_stability
      temperature_from_potential_temperature
      thickness_hydrostatic
      weighted_continuous_average


Moist Thermodynamics
--------------------

   .. autosummary::
      :toctree: ./

      dewpoint
      dewpoint_from_relative_humidity
      dewpoint_from_specific_humidity
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
      scale_height
      specific_humidity_from_dewpoint
      specific_humidity_from_mixing_ratio
      thickness_hydrostatic_from_relative_humidity
      vapor_pressure
      vertical_velocity
      vertical_velocity_pressure
      virtual_potential_temperature
      virtual_temperature
      virtual_temperature_from_dewpoint
      wet_bulb_temperature
      wet_bulb_potential_temperature


Soundings
---------

   .. autosummary::
      :toctree: ./

      bulk_shear
      bunkers_storm_motion
      corfidi_storm_motion
      cape_cin
      ccl
      critical_angle
      cross_totals
      downdraft_cape
      el
      galvez_davison_index
      k_index
      lcl
      lfc
      lifted_index
      mixed_layer
      mixed_layer_cape_cin
      mixed_parcel
      most_unstable_cape_cin
      most_unstable_parcel
      parcel_profile
      parcel_profile_with_lcl
      parcel_profile_with_lcl_as_dataset
      showalter_index
      significant_tornado
      storm_relative_helicity
      supercell_composite
      surface_based_cape_cin
      sweat_index
      total_totals_index
      vertical_totals


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
      rotational_wind_from_inversion
      divergent_wind_from_inversion

Boundary Layer/Turbulence
-------------------------

   .. autosummary::
      :toctree: ./

      brunt_vaisala_frequency
      brunt_vaisala_frequency_squared
      brunt_vaisala_period
      friction_velocity
      gradient_richardson_number
      tke


Mathematical Functions
----------------------

   .. autosummary::
      :toctree: ./

      cross_section_components
      first_derivative
      geospatial_gradient
      geospatial_laplacian
      gradient
      laplacian
      lat_lon_grid_deltas
      normal_component
      second_derivative
      tangential_component
      unit_vectors_from_cross_section
      vector_derivative


Apparent Temperature
--------------------

   .. autosummary::
      :toctree: ./

      apparent_temperature
      heat_index
      windchill

Standard Atmosphere
-------------------

   .. autosummary::
      :toctree: ./

      altimeter_to_sea_level_pressure
      altimeter_to_station_pressure
      height_to_pressure_std
      pressure_to_height_std

Smoothing
---------
   .. autosummary::
      :toctree: ./

      smooth_gaussian
      smooth_window
      smooth_rectangular
      smooth_circular
      smooth_n_point
      zoom_xarray

Other
-----

   .. autosummary::
      :toctree: ./

      angle_to_direction
      azimuth_range_to_lat_lon
      bounding_box_mask
      find_bounding_indices
      find_bounding_box_indices
      find_intersections
      get_layer
      get_layer_heights
      get_perturbation
      get_vectorized_array_indices
      isentropic_interpolation
      isentropic_interpolation_as_dataset
      nearest_intersection_idx
      parse_angle
      reduce_point_density
      resample_nn_1d
