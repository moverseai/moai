
class Constants(object):
    
    # ACTIONS    
    _ACTION_TEST_ = 'test'
    _ACTION_VAL_ = 'val'
    _ACTION_PREDICT_ = 'predict'
    _ACTION_FIT_ = 'fit'
    _ACTION_RESUME_ = 'resume'

    # MOAI EXECUTION
    _MOAI_: str = '_moai_'

    # _MOAI_MONITORING_: str = f'{_MOAI_}._monitoring_'
    _MOAI_METRICS_: str = f'{_MOAI_}._metrics_'
    _MOAI_LOSSES_: str = f'{_MOAI_}._losses_'
    _MOAI_LOSSES_RAW_: str = f'{_MOAI_}._losses_.raw'
    _MOAI_LOSSES_WEIGHTED_: str = f'{_MOAI_}._losses_.weighted'
    _MOAI_LOSSES_TOTAL_: str = f'{_MOAI_}._losses_.total'
    # _MOAI_EXECUTION_: str = f'{_MOAI_}._execution_'
    # _MOAI_INITIALIZE_: str = f'{_MOAI_}._initialize_'

    _EXECUTION_: str = f'_execution_'
    
    _LIGHTNING_STEP_: str = f'_lightning_step_'
    _MOAI_EXECUTION_LIGHTNING_STEP_: str = f'{_MOAI_}.{_EXECUTION_}.{_LIGHTNING_STEP_}'
    _MOAI_EXECUTION_MONITORING_: str = f'{_MOAI_}.{_EXECUTION_}._monitoring_'
    _MOAI_EXECUTION_INITIALIZE_: str = f'{_MOAI_}.{_EXECUTION_}._initialize_'
    
    _EXECUTION_LIGHTNING_STEP_: str = f'{_EXECUTION_}.{_LIGHTNING_STEP_}'
    _EXECUTION_MONITORING_: str = f'{_EXECUTION_}._monitoring_'
    _SCHEDULE_: str = f'_schedule_'    
    _EXECUTION_SCHEDULE_: str = f'{_EXECUTION_}._schedule_'
    _INITIALIZE_: str = f'_initialize_'
    _EXECUTION_INITIALIZE_: str = f'{_EXECUTION_}._initialize_'

    _OPTIMIZATION_STEP_: str = f'_optimization_step_'

    _COLLECTIONS_: str = f'_collections_'
    _MONITORS_: str = f'_monitors_'
    _METRICS_: str = f'_metrics_'
    _FLOWS_: str = f'_flows_'
    _OPTIMIZER_GROUPS_: str = f'_groups_'
    
    _DEFINITIONS_: str = f'_definitions_'
    _METRICS_COLLECTION_: str = f'{_DEFINITIONS_}.{_COLLECTIONS_}._metrics_'
    _OBJECTIVES_COLLECTION_: str = f'{_DEFINITIONS_}.{_COLLECTIONS_}._objectives_'
    _MONITORS_COLLECTION_: str = f'{_DEFINITIONS_}.{_COLLECTIONS_}._monitors_'
    _CRITERIA_COLLECTION_: str = f'{_DEFINITIONS_}.{_COLLECTIONS_}._criteria_'
    _OPTIMIZERS_COLLECTION_: str = f'{_DEFINITIONS_}.{_COLLECTIONS_}._optimizers_'
    _SCHEDULERS_COLLECTION_: str = f'{_DEFINITIONS_}.{_COLLECTIONS_}._schedulers_'
    _DEFINED_FLOWS_: str = f'{_DEFINITIONS_}._flows_'
    
    
    _STAGE_: str = f'_stage_'

    _PARAMS_: str = f'_params_'
    _OUT_: str = f'_out_' # f'out'
    _WEIGHT_: str = f'_weight_'
    _REDUCTION_: str = f'_reduction_'
    _FIT_: str = f'_fit_'
    _VAL_: str = f'_val_'
    _TEST_: str = f'_test_'
    _PREDICT_: str = f'_predict_'
    _BATCH_: str = f'_batch_'
    _FREQUENCY_: str = f'_frequency_'
    _ITERATIONS_: str = f'_iterations_'
    _REFRESH_OPTIMIZERS_: str = f'_refresh_optimizers_'    
    _OBJECTIVE_: str = f'_objective_'
    _ASSIGN_: str = f'_assign_'
    _OPTIMIZER_: str = f'_optimizer_'
    _TERMINATION_: str = f'_termination_'
    _SETUP_: str = f'_setup_'
    _BATCH_: str = f'_batch_'
    _EPOCH_: str = f'_epoch_'
    _INTERVAL_: str = f'_interval_'
    _TYPE_: str = f'_type_'

    _SCHEDULE_FIT_: str = f'_fit_'
    _SCHEDULE_VAL_: str = f'_val_'

    # MODEL
    _MODEL_: str = 'model'

    _PARAMETERS_: str = 'parameters'
    _CRITERIA_: str = f'criteria'

    _PARAMETER_GROUPS_: str = f'{_PARAMETERS_}.groups'
    _PARAMETER_OPTIMIZERS_: str = f'{_PARAMETERS_}.optimizers'
    _PARAMETER_INITIALIZERS_: str = f'{_PARAMETERS_}.initializers'
    _PARAMETERS_CRITERIA_: str = f'{_PARAMETERS_}.criteria'
    _PARAMETER_SCHEDULERS_: str = f'{_PARAMETERS_}.schedulers'
    _OPTIMIZERS_: str = f'optimizers'
    _INITIALIZERS_: str = f'initializers'    
    _CRITERIA_: str = f'criteria'
    _GROUPS_: str = f'groups'
    _SCHEDULERS_: str = f'schedulers'
    _OBJECTIVES_: str = f'objectives'    
    # _MODEL_METRICS_: str = f'{_MODEL_}.metrics'
    # _MODEL_MONITORS_: str = f'{_MODEL_}.monitors'
    # _MODEL_MODULES_: str = f'{_MODEL_}.modules'    
    # _MODEL_MONADS_: str = f'{_MODEL_}.monads'
    # _MODEL_REMODEL_: str = f'{_MODEL_}.remodel'

    # ENGINE
    _ENGINE_: str = 'engine'

    _RUNNER_: str = f'{_ENGINE_}.runner'
    _MODULES_: str = f'{_ENGINE_}.modules'
    _LOGGERS_: str = f'{_ENGINE_}.loggers' # or directly in runner?

    # DATA
    _DATA_: str = 'data'

    _TRAIN_DATA_: str = f'{_DATA_}.train'
    _VALIDATION_DATA_: str = f'{_DATA_}.val'
    _TEST_DATA_: str = f'{_DATA_}.test'
    _PREDICT_DATA_: str = f'{_DATA_}.predict'
