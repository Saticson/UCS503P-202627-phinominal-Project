package com.axiomcloud.repository;

import com.axiomcloud.model.MarketplaceListingEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface MarketplaceListingRepository extends JpaRepository<MarketplaceListingEntity, String> {

    List<MarketplaceListingEntity> findAllByOrderByMetricScoreDesc();

    List<MarketplaceListingEntity> findByModalityIgnoreCaseOrderByMetricScoreDesc(String modality);

    List<MarketplaceListingEntity> findByVerifiedTrueOrderByMetricScoreDesc();

    List<MarketplaceListingEntity> findByModalityIgnoreCaseAndVerifiedTrueOrderByMetricScoreDesc(String modality);

    @Query("SELECT m FROM MarketplaceListingEntity m WHERE " +
           "(:modality IS NULL OR :modality = 'ALL' OR UPPER(m.modality) = UPPER(:modality)) AND " +
           "(:verifiedOnly IS NULL OR :verifiedOnly = false OR m.verified = true) AND " +
           "(:query IS NULL OR :query = '' OR " +
           "LOWER(m.title) LIKE LOWER(CONCAT('%', :query, '%')) OR " +
           "LOWER(m.description) LIKE LOWER(CONCAT('%', :query, '%')) OR " +
           "LOWER(m.tagsCsv) LIKE LOWER(CONCAT('%', :query, '%'))) " +
           "ORDER BY m.metricScore DESC")
    List<MarketplaceListingEntity> searchListings(
            @Param("modality") String modality,
            @Param("query") String query,
            @Param("verifiedOnly") Boolean verifiedOnly);

    List<MarketplaceListingEntity> findTop4ByModalityIgnoreCaseOrderByMetricScoreDesc(String modality);
}
