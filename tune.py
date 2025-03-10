from fusion import LowRankFusion, GatedFusion, TransformerFusion, EarlyFusion, Config, Driver


from pathlib import Path
from itertools import combinations, product
import constants

textual = constants.SAVED_EMBEDDINGS_DIR / "textual"
relational = constants.SAVED_EMBEDDINGS_DIR / "relational"

textual_embeddings = textual.glob("**/*.pt")
relational_embeddings = relational.glob("**/*.pt")

search_space = {
    "input_features": list(product(textual_embeddings, relational_embeddings)),
    "latent_dim": [16, 32, 64, 128],
    "model_cls": [LowRankFusion, GatedFusion, TransformerFusion, EarlyFusion],
}


for latent_dim in search_space['latent_dim']:
    for model in search_space['model_cls']:
        for t, r in search_space['input_features']:
            tdim, tname = t.stem, t.parent.stem
            rdim, rname = r.stem, r.parent.stem
            config = Config(
                model_cls=EarlyFusion,
                textual_path=t,
                relational_path=r,
                latent_dim=latent_dim,
                max_epochs=1,
                lr=0.01,
                seed=0,
                prefix=f"{tname}_{rname}/{tdim}_{rdim}",
                kwargs={}
            )
            driver = Driver(config)
            driver.run()