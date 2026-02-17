import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer.gan_trainer import GANTrainer
from src.utils.device import resolve_device
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training HiFi-GAN. Instantiates generator, discriminators,
    optimizers, schedulers, metrics, logger, writer, and dataloaders.
    Runs GANTrainer to train and evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    device = resolve_device(config.trainer.device)

    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(
        config, text_encoder=None, device=device
    )

    # build model architecture
    generator = instantiate(config.generator).to(device)
    mpd = instantiate(config.discriminator.mpd).to(device)
    msd = instantiate(config.discriminator.msd).to(device)

    logger.info("Generator:")
    logger.info(generator)
    logger.info("MPD:")
    logger.info(mpd)
    logger.info("MSD:")
    logger.info(msd)

    # Собираем дискриминаторы в словарь
    discriminators = {
        "mpd": mpd,
        "msd": msd,
    }

    # get loss functions (criterion - это словарь с generator и discriminator лоссами)
    criterion = {
        "generator": instantiate(config.loss.generator).to(device),
        "discriminator": instantiate(config.loss.discriminator).to(device),
    }

    # metrics
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    # build optimizers (отдельно для генератора и дискриминаторов)
    optimizer_g = instantiate(config.optimizer.generator, params=generator.parameters())
    optimizer_d = instantiate(
        config.optimizer.discriminator,
        params=list(mpd.parameters()) + list(msd.parameters()),
    )

    # learning rate schedulers
    lr_scheduler_g = instantiate(config.lr_scheduler.generator, optimizer=optimizer_g)
    lr_scheduler_d = instantiate(
        config.lr_scheduler.discriminator, optimizer=optimizer_d
    )

    # mel-spectrogram transform для валидации
    mel_spec_transform = instantiate(config.mel_spec_transform).to(device)

    # epoch_len для iteration-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = GANTrainer(
        generator=generator,
        discriminators=discriminators,
        criterion=criterion,
        metrics=metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        mel_spec_transform=mel_spec_transform,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        epoch_len=epoch_len,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
