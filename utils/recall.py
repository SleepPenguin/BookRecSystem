from .match import Annoy

def topn_evaluate(trainer, user_dl, user_set, item_dl, item_set, user_col, item_col, model, model_dir, topk):
    print("generated user embedding and item embedding")
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=user_dl, model_path=model_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=model_dir)
    annoy = Annoy(n_trees=100)
    annoy.fit(item_embedding, model_dir)
    n_total = 0
    n_hit = 0
    for true_item_list, user_emb in zip(user_set['pos_list'], user_embedding):
        n_total += 1
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)  # the index of topk match items
        for recall_item  in item_set[item_col][items_idx]:
            if recall_item in true_item_list:
                n_hit += 1
    hr = n_hit/n_total
    topk_score = {'HR': hr, 'n_hit': n_hit, 'n_total': n_total}
    print(topk_score)
    return user_embedding, item_embedding, topk_score