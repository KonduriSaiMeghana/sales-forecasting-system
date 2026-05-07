import pandas as pd


def fill_missing_dates(df):

    final_df = []

    for state in df['state'].unique():

        state_df = df[df['state'] == state].copy()

        state_df = state_df.set_index('date')

        # Create continuous dates
        idx = pd.date_range(
            start=state_df.index.min(),
            end=state_df.index.max(),
            freq='D'
        )

        state_df = state_df.reindex(idx)

        state_df['state'] = state

        # Fill missing sales using interpolation
        state_df['sales'] = state_df['sales'].interpolate(method='linear')
        
        # Fill any remaining NaN at the start/end
        state_df['sales'] = state_df['sales'].ffill().bfill()

        state_df = state_df.reset_index()

        state_df.rename(columns={'index': 'date'}, inplace=True)

        final_df.append(state_df)

    final_df = pd.concat(final_df, ignore_index=True)

    return final_df