Modules
1) Data
    1. Obtain data from Kaggle
        A. Download
    2. Transform data
        A. Clean
        B. Put into appropriate task-specific format
    3. Split data
        A. Train
        B. Validate
        C. Test
2) Run
    0. Initialization
        - Have defaults
    1. Train
        A. Prepare directories
        B. Build hooks
        C. Build model
        D. Build loss
        E. Build optimizer
        F. Load checkpoint
        G. Build scheduler
        H. Build dataloader/dataset
        I. Build summary writer
        J. Run train loop
            for epoch in range(last_epoch, config.train.num_epochs)
                # Train
                for dataloader in dataloaders:
                    split = dataloader['split']
                    dataset_mode = dataloader['mode']
                    if dataset_mode != 'train':
                        continue
                    dataloader = dataloader['dataloader']
                    train_single_epoch(config, model, split, dataloader, hooks, optimizer, scheduler, epoch)
                # Validation
                score_dict = {}
                checkpoint_score = None
                for dataloader in dataloaders:
                    split = dataloader['split']
                    dataset_mode = dataloader['mode']
                    if dataset_mode != 'validation':
                        continue
                    dataloader = dataloader['dataloader']
                    score = evaludate_single_epoch(config, model, split, dataloader, hooks, optimizer, scheduler, epoch)
                    score_dict[split] = score
                    if checkpoint_score is None:
                        checkpoint_score = score
                # Update Learning Rates
                # Calling scheduler.step(**kwargs)
                if config.scheduler.name == 'ReduceLROnPlateau':
                    scheduler.step(checkpoint_score)
                elif config.scheduler.name == 'CosineAnnealingLR':
                    param_epoch = (epoch + 1) % config.scheduler.params.T_max
                    scheduler.step(param_epoch + 1)
                elif config.scheduler.name != 'OneCycleLR' and config.scheduler.name != 'ReduceLROnPlateau':
                    scheduler.step()
                
                # Checkpointing
                if checkpoint_score > best_checkpoint_score:
                    best_checkpoint_score = checkpoint_score
                    knlp.utils.save_checkpoint(config, model, optimizer, epoch, keep=None, name='best.score')
                    knlp.utils.copy_last_n_checkpoints(config, 5, 'best.score.{:04d}.pth')
                if epoch % config.train.save_checkpoint_epoch == 0:
                    knlp.utils.save_checkpoint(config, model, optimizer, epoch, keep=config.train.num_keep_checkpoint)
    2. Evaluate
    3. Inference
    4. SWA*
3) Submit


Configurations
    - Dataset
        - Name
        - Path
        - Splits
    - Transform
    - Model
    - Train
    - Evaluate
    - Optimizer
    - Scheduler
    - Hooks
        - Metric
        - Loss
        - Build Model
        - Post Forward
        - Write Result