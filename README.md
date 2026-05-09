## Dataset

This project uses four processed recommendation datasets: MovieLens-1M, Amazon Books, Amazon Electronics, and Book-Crossing. All datasets are converted into a unified format to support consistent training, evaluation, and comparison across domains.

Each dataset contains three files:

- `users.csv`: user profile information with `user_id`, `name`, `gender`, `age`, and `status`
- `items.csv`: item metadata with `item_id`, `title`, `genre`, and `description`
- `ratings.csv`: user-item interactions with `user_id`, `item_id`, and `rating`

This unified structure makes the datasets suitable for recommender-system experiments, user-behaviour simulation, and cross-domain evaluation.

## Operating CtrlAgent

`CtrlAgent` is the main user-behaviour simulation framework used in this project. It stores and manages each user's profile, memory, and action history.
